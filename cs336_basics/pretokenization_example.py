import os
from typing import BinaryIO
import multiprocessing
import regex as re
import collections
import time
import tracemalloc
import pickle
import re as resplit


# ds = load_dataset("Skylion007/openwebtext")
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  
        while True:
            mini_chunk = file.read(mini_chunk_size)  

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def worker_task(input_path, start, end, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    mathcount = {}
    text = open(input_path, "rb")
    text.seek(start)
    chunk = text.read(end - start).decode("utf-8", errors="ignore")
    chunk = [chunk] 
            
    for special_token in special_tokens:
        new_segment = []
        for segment in chunk:
            piece = segment.split(special_token)
            new_segment.extend(piece)
        chunk = new_segment


    for segment in chunk:
        mathes = re.finditer(PAT, segment)
        for match in mathes:
            token = match.group()
            if token not in mathcount:
                mathcount[token] = 1
            else:
                mathcount[token] = mathcount[token] + 1
    text.close()

    return mathcount

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], max_workers: int = 4):
    vocab = {i: bytes([i]) for i in range(256)}
    new_token_id = 256
    
    for special_token in special_tokens:
        vocab[new_token_id] = special_token.encode("utf-8")
        new_token_id += 1

    # بررسی حجم فایل برای انتخاب رویکرد (مثلاً آستانه 1 مگابایت)
    text = open(input_path, "rb")
    text.seek(0, os.SEEK_END)
    file_size = text.tell()
    text.seek(0)
    
    threshold_bytes = 1024 * 1024  # 1 MB
    use_multiprocessing = (file_size > threshold_bytes) and (max_workers > 1)
    
    # ---------------------------------------------------------
    # 1. Chunk Boundaries Time
    # ---------------------------------------------------------
    # tracemalloc.start()
    t0 = time.perf_counter()
    
    split_token = special_tokens[-1].encode("utf-8") if special_tokens else b"\n"
    
    if use_multiprocessing:
        boundaries = find_chunk_boundaries(text, max_workers, split_token)
    else:
        boundaries = [0, file_size]
    
    text.close()
    t1 = time.perf_counter()
    print(f"[Timer] 1. Boundaries finding: {t1 - t0:.4f}s")

    # ---------------------------------------------------------
    # 2. Pre-tokenization & Counting Time
    # ---------------------------------------------------------
    t2 = time.perf_counter()
    if use_multiprocessing:
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end, special_tokens))

        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(worker_task, tasks)
        
        merged_mathcount = collections.Counter()
        for res in results:
            merged_mathcount.update(res)
        mathcount = dict(merged_mathcount)
    else:
        # حالت Serial (فایل کوچک یا max_workers=1)
        mathcount = worker_task(input_path, 0, file_size, special_tokens)
        
    t3 = time.perf_counter()
    mode = f"Multiprocessing ({max_workers} cores)" if use_multiprocessing else "Serial"
    print(f"[Timer] 2. Pre-tokenization ({mode}): {t3 - t2:.4f}s")

    # ---------------------------------------------------------
    # 3. BPE Merge Loop Time
    # ---------------------------------------------------------
    t4 = time.perf_counter()
    current_sequences = {i: list(i.encode("utf-8")) for i in mathcount}
    pair_locations = collections.defaultdict(set)

    merges = []
    global_frequency = {}
    for pre_token in mathcount:
            sequence = current_sequences[pre_token]
            count = mathcount[pre_token]
            for x in range(len(sequence)-1):
                bigram = sequence[x], sequence[x+1]
                if bigram not in global_frequency:
                    global_frequency[bigram] = count
                else:
                    global_frequency[bigram] += count
                pair_locations[bigram].add(pre_token)

    while new_token_id < vocab_size:

        if not global_frequency:
            break

        token = max(global_frequency, key=lambda pair: (global_frequency[pair], (vocab[pair[0]], vocab[pair[1]])))
        
        vocab[new_token_id] = vocab[token[0]] + vocab[token[1]]
        byte_merged = vocab[token[0]], vocab[token[1]]
        merges.append(byte_merged)

        affected_tokens = list(pair_locations[token])

        for pre_token in affected_tokens:
            sequence = current_sequences[pre_token]

            for x in range(len(sequence)-1):
                bg = sequence[x], sequence[x+1]
                global_frequency[bg] -= mathcount[pre_token]
                if global_frequency[bg] == 0:
                    del global_frequency[bg]  
                if pre_token in pair_locations[bg]:
                    pair_locations[bg].remove(pre_token)

            merged_seq = []
            index = 0
            
            while index < len(sequence):
                if index < len(sequence) - 1 and sequence[index] == token[0] and sequence[index+1] == token[1]:
                    merged_seq.append(new_token_id)
                    index += 2
                else:
                    merged_seq.append(sequence[index])
                    index += 1

            current_sequences[pre_token] = merged_seq
            for x in range(len(merged_seq)-1):
                bg = merged_seq[x], merged_seq[x+1]
                if bg not in global_frequency:
                    global_frequency[bg] = mathcount[pre_token]
                else:
                    global_frequency[bg] += mathcount[pre_token]
                pair_locations[bg].add(pre_token)

        new_token_id += 1

    t5 = time.perf_counter()
    print(f"[Timer] 3. BPE Merging loop: {t5 - t4:.4f}s")
    print(f"[Timer] Total Time: {t5 - t0:.4f}s\n")
    # print(f"[Memory] Total memory used: {tracemalloc.get_traced_memory()}")
    # tracemalloc.stop()

    return vocab, merges

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        self.ids = {v: k for k, v in vocab.items()}
        for special_token in self.special_tokens:
            bspecial_token = special_token.encode('utf-8')
            if (bspecial_token in self.ids):
                continue
            else:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = bspecial_token
                self.ids[bspecial_token] = new_id
        
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}


    @classmethod
    def from_files(cls, serialized_path, special_tokens=None):
        with open(serialized_path, 'rb') as f:
            vocab, merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text): # it should return a list of integers # 1. Split text around special tokens, preserving special tokens # 2. Pre-tokenize each normal-text segment with the training regex. # 3. Convert each pre-token to UTF-8 byte-level tokens. # 4. Apply learned BPE merges independently within each pre-token, #    respecting merge priority. # 5. Map final byte tokens to vocabulary IDs. # 6. Insert special-token IDs at their original positions. # 7. Return the full list of token IDs.
        # I imported re as resplit
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text = [text]
        sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)
        for special_token in sorted_special_tokens:
                pattern = "(" + resplit.escape(special_token) + ")"
                new_segment = []
                for segment in text:
                    if segment in self.special_tokens:
                        new_segment.append(segment)
                    else: 
                        piece = resplit.split(pattern, segment)
                        new_segment.extend(piece)
                text = new_segment
        
        final_output_ids = []

        for segment in text:
            if segment in self.special_tokens:
                segment = segment.encode("utf-8")
                tokenID = self.ids[segment]
                final_output_ids.append(tokenID)
            else:
                matches = re.finditer(PAT, segment)
                for match in matches:
                    sequence= []
                    pre_token = match.group()
                    encoded_pre_tokens = pre_token.encode("utf-8")
                    for j in encoded_pre_tokens:
                        byte_token = bytes([j])
                        sequence.append(self.ids[byte_token])
                    while len(sequence) >= 2:
                        best_pair = None
                        best_rank = float('inf')
                        best_byte_pair = None
                        
                        for i in range(len(sequence) - 1):
                            pair = (sequence[i], sequence[i+1])
                            byte_pair = (self.vocab[pair[0]], self.vocab[pair[1]])
                            
                            if byte_pair in self.merge_ranks:
                                rank = self.merge_ranks[byte_pair]
                                if rank < best_rank:
                                    best_rank = rank
                                    best_pair = pair
                                    best_byte_pair = byte_pair

                        
                        if best_pair is None:
                            break

                        new_token_id = self.ids[best_byte_pair[0]+best_byte_pair[1]] 
                        
                        new_sequence = []
                        i = 0
                        while i < len(sequence):
                            if i < len(sequence) - 1 and sequence[i] == best_pair[0] and sequence[i+1] == best_pair[1]:
                                new_sequence.append(new_token_id)
                                i += 2  
                            else:
                                new_sequence.append(sequence[i])
                                i += 1  
                        
                        sequence = new_sequence
                    
                    final_output_ids.extend(sequence)
        return final_output_ids

    def encode_iterable(self, iterable):
        buffer = ''
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)

        for segment in iterable:
            text = buffer + segment
            if not text:
                continue
                
            # ۱. پیدا کردن طولانی‌ترین suffix مشکوک به special token
            max_suffix_len = 0
            for sp in sorted_special_tokens:
                # تغییر مهم: بررسی تا (طول توکن منهای یک) برای جلوگیری از حبس توکن‌های کامل
                for i in range(min(len(text), len(sp) - 1), 0, -1):
                    if text.endswith(sp[:i]):
                        max_suffix_len = max(max_suffix_len, i)
                        break
            
            # ۲. جدا کردن متن نسبتاً امن و پسوند مشکوک
            if max_suffix_len > 0:
                safe_text = text[:-max_suffix_len]
                suspicious_suffix = text[-max_suffix_len:]
            else:
                safe_text = text
                suspicious_suffix = ''
                
            # اگر متن امنی نداریم، همه چیز را در بافر نگه دار برای قطعه‌ی بعدی
            if not safe_text:
                buffer = suspicious_suffix
                continue

            # ۳. استخراج special token های *کامل* از داخل safe_text
            parts = [safe_text]
            for sp in self.special_tokens:
                pattern = "(" + resplit.escape(sp) + ")"
                new_parts = []
                for p in parts:
                    new_parts.extend(resplit.split(pattern, p))
                parts = [p for p in new_parts if p]

            # ۴. اعمال Regex و ساخت بافر جدید
            # ۴. اعمال Regex و ساخت بافر جدید
            # اختصاص مستقیم suffix مشکوک به عنوان پایه بافر جدید
            new_buffer = suspicious_suffix 
            
            for i, part in enumerate(parts):
                if part in self.special_tokens:
                    yield from self.encode(part)
                else:
                    matches = list(re.finditer(PAT, part))
                    if not matches:
                        continue
                        
                    if i == len(parts) - 1:
                        last_match = matches[-1]
                        # ترکیب آخرین توکن متنی با پسوند مشکوک
                        new_buffer = last_match.group() + suspicious_suffix
                        
                        for match in matches[:-1]:
                            yield from self.encode(match.group())
                    else:
                        for match in matches:
                            yield from self.encode(match.group())
                            
            # آپدیت کردن بافر برای دور بعدی حلقه
            buffer = new_buffer
        # ۵. در پایان کل استریم، اگر چیزی در بافر مانده (که قطعاً special token نبوده) آن را encode می‌کنیم
        if buffer:
            yield from self.encode(buffer)


    def decode(self, ids):
        sequence = []
        for i in ids:
            sequence.append(self.vocab[i])
        result = b"".join(sequence)
        result = result.decode("utf-8", errors='replace')
        return result




if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    path = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/owt_valid.txt"
    serialized_path = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/owt_valid.pkl"
    if os.path.exists(path):
        if os.path.exists(serialized_path):
            print(f"Serialized data is importing...")
            with open(serialized_path, "rb") as f:
                vocab, merges = pickle.load(f)
            max_key, max_value = max(vocab.items(), key = lambda x: len((x[1])))
            print("Token ID:", max_key)
            print("Bytes:", max_value)
            print("Length in bytes:", len(max_value))
            print("Decoded:", max_value.decode("utf-8", errors="replace"))
        else:
            print("Running benchmark on main dataset...")
            workers = 8
            print(f"\n--- Testing with {workers} workers ---")
            vocab, merges = train_bpe(path, 10000, special_tokens, max_workers=workers)
            with open(serialized_path, "wb") as fw:
                pickle.dump([vocab, merges], fw)
            print(f"Run Success! Total Vocab: {(vocab)}, Total Merges: {(merges)}")




