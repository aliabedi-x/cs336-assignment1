import os
import time
import pickle
import multiprocessing
import collections
import regex as re
import gc
import heapq
from functools import partial
from datasets import load_dataset

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# ---------------------------------------------------------
# ساختار سفارشی برای تبدیل Min-Heap پایتون به Max-Heap
# ---------------------------------------------------------
class MaxHeapEntry:
    __slots__ = ['freq', 'pair', 'v1', 'v2']
    def __init__(self, freq, pair, v1, v2):
        self.freq = freq
        self.pair = pair
        self.v1 = v1
        self.v2 = v2

    def __lt__(self, other):
        # اولویت اول: فرکانس بالاتر (علامت بزرگتر برای معکوس کردن Min-Heap)
        if self.freq != other.freq:
            return self.freq > other.freq
        # اولویت دوم (Tie-breaker): بایت‌های از نظر لغوی بزرگ‌تر
        if self.v1 != other.v1:
            return self.v1 > other.v1
        return self.v2 > other.v2

def worker_task_hf(batch_texts, special_tokens):
    local_counts = collections.Counter()
    for text in batch_texts:
        segments = [text]
        for st in special_tokens:
            new_segments = []
            for seg in segments:
                new_segments.extend(seg.split(st))
            segments = new_segments
            
        for seg in segments:
            for match in PAT.finditer(seg):
                local_counts[match.group()] += 1
    return local_counts

def text_batch_generator(dataset, batch_size=1000):
    batch = []
    for item in dataset:
        batch.append(item["text"])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def train_bpe_optimized(dataset, vocab_size: int, special_tokens: list[str], max_workers: int = 8):
    vocab = {i: bytes([i]) for i in range(256)}
    new_token_id = 256
    
    for st in special_tokens:
        vocab[new_token_id] = st.encode("utf-8")
        new_token_id += 1

    t0 = time.perf_counter()
    print("[Phase 1] Starting Map-Reduce Pre-tokenization...")
    global_mathcount = collections.Counter()
    
    batches = text_batch_generator(dataset, batch_size=1000)
    worker = partial(worker_task_hf, special_tokens=special_tokens)
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(worker, batches)
        for i, local_count in enumerate(results):
            global_mathcount.update(local_count)
            if (i + 1) % 50 == 0:
                print(f"  ... processed {(i + 1) * 1000} documents")

    t1 = time.perf_counter()
    print(f"[Timer] Phase 1 (Pre-tokenization): {t1 - t0:.4f}s")
    print(f"  Total unique pre-tokens found: {len(global_mathcount):,}")

    t2 = time.perf_counter()
    print("\n[Phase 2] Building ID-based Indexing & Initializing Priority Queue...")
    
    unique_words = list(global_mathcount.keys())
    word_freqs = []
    word_splits = []
    
    for word in unique_words:
        word_freqs.append(global_mathcount[word])
        word_splits.append(list(word.encode("utf-8")))

    del global_mathcount
    del unique_words
    gc.collect()
        
    pair_locations = collections.defaultdict(set)
    global_frequency = collections.Counter()

    for word_id, sequence in enumerate(word_splits):
        count = word_freqs[word_id]
        for x in range(len(sequence) - 1):
            bg = (sequence[x], sequence[x+1])
            global_frequency[bg] += count
            pair_locations[bg].add(word_id)

    # ساخت Heap اولیه
    pq = []
    for pair, freq in global_frequency.items():
        heapq.heappush(pq, MaxHeapEntry(freq, pair, vocab[pair[0]], vocab[pair[1]]))

    t3 = time.perf_counter()
    print(f"[Timer] Phase 2 (Indexing): {t3 - t2:.4f}s")

    t4 = time.perf_counter()
    print(f"\n[Phase 3] Starting Delta BPE Merge Loop (Target Vocab: {vocab_size})...")
    merges = []
    
    time_winner_selection = 0.0
    time_affected_update = 0.0

    while new_token_id < vocab_size:
        # -----------------------------------------------------
        # A: Winner Selection (Priority Queue + Lazy Invalidation)
        # -----------------------------------------------------
        t_w_start = time.perf_counter()
        
        token = None
        while pq:
            entry = heapq.heappop(pq)
            current_freq = global_frequency.get(entry.pair, 0)
            
            # Lazy Invalidation: اگر فرکانس فعلی با فرکانس داخل رکورد Heap یکی نبود، آن را دور بینداز
            if current_freq != entry.freq:
                continue
                
            token = entry.pair
            break
            
        if not token:
            break # Heap خالی شده یا تمام Pair ها نامعتبر بوده‌اند

        vocab[new_token_id] = vocab[token[0]] + vocab[token[1]]
        byte_merged = vocab[token[0]], vocab[token[1]]
        merges.append(byte_merged)
        
        t_w_end = time.perf_counter()
        time_winner_selection += (t_w_end - t_w_start)

        # -----------------------------------------------------
        # B: Updating Affected Words
        # -----------------------------------------------------
        t_u_start = time.perf_counter()
        
        affected_word_ids = list(pair_locations[token])
        affected_pairs_in_this_merge = set()

        for word_id in affected_word_ids:
            sequence = word_splits[word_id]
            count = word_freqs[word_id]

            # حذف جفت‌های قدیمی
            for x in range(len(sequence) - 1):
                bg = (sequence[x], sequence[x+1])
                global_frequency[bg] -= count
                affected_pairs_in_this_merge.add(bg)
                
                if global_frequency[bg] == 0:
                    del global_frequency[bg]  
                if word_id in pair_locations[bg]:
                    pair_locations[bg].remove(word_id)

            # انجام Merge
            merged_seq = []
            index = 0
            while index < len(sequence):
                if index < len(sequence) - 1 and sequence[index] == token[0] and sequence[index+1] == token[1]:
                    merged_seq.append(new_token_id)
                    index += 2
                else:
                    merged_seq.append(sequence[index])
                    index += 1

            word_splits[word_id] = merged_seq
            
            # ثبت جفت‌های جدید
            for x in range(len(merged_seq) - 1):
                bg = (merged_seq[x], merged_seq[x+1])
                global_frequency[bg] += count
                affected_pairs_in_this_merge.add(bg)
                pair_locations[bg].add(word_id)

        # تزریق یکباره مقادیر جدید به Heap برای جلوگیری از تورم سایز صف
        for bg in affected_pairs_in_this_merge:
            if bg in global_frequency:
                heapq.heappush(pq, MaxHeapEntry(global_frequency[bg], bg, vocab[bg[0]], vocab[bg[1]]))

        new_token_id += 1
        
        t_u_end = time.perf_counter()
        time_affected_update += (t_u_end - t_u_start)
        
        if len(merges) % 2000 == 0:
            print(f"  ... {len(merges)} merges completed (Current Vocab: {new_token_id})")

    t5 = time.perf_counter()
    print(f"\n[Profiler] Winner Selection Time: {time_winner_selection:.4f}s")
    print(f"[Profiler] Affected Updates Time: {time_affected_update:.4f}s")
    print(f"[Timer] Phase 3 Total Time: {t5 - t4:.4f}s")
    print(f"[Timer] Total Execution Time: {t5 - t0:.4f}s\n")

    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    serialized_path = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/owt_train.pkl"

    if os.path.exists(serialized_path):
        print("Serialized data is importing...")
        with open(serialized_path, "rb") as f:
            vocab, merges = pickle.load(f)
        max_key, max_value = max(vocab.items(), key=lambda x: len((x[1])))
        print("Token ID:", max_key)
        print("Decoded:", max_value.decode("utf-8", errors="replace"))
    else:
        print("Loading local Text file...")
        txt_file_path = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/owt_train.txt"
        
        ds = load_dataset(
            "text",  # تغییر فرمت به متن
            data_files=txt_file_path,  # مسیر مستقیم فایل txt
            split="train", 
            streaming=True
        )
        
        ds_subset = ds # .take(2000000)
        
        vocab, merges = train_bpe_optimized(
            dataset=ds_subset, 
            vocab_size=32000, 
            special_tokens=special_tokens, 
            max_workers=8
        )
        
        with open(serialized_path, "wb") as fw:
            pickle.dump([vocab, merges], fw)
            
        print(f"Run Success! Total Vocab: {len(vocab)}, Total Merges: {len(merges)}")