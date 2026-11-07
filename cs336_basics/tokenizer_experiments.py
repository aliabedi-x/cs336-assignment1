from pretokenization_example import Tokenizer
import pickle
import random
import time

def compression_ratio(tokenizer, documents):
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        num_bytes = len(document.encode("utf-8"))
        num_tokens = len(tokenizer.encode(document))
        total_bytes += num_bytes
        total_tokens += num_tokens
    comp_ratio = total_bytes/total_tokens
    return comp_ratio

def iter_documents(path, separator="<|endoftext|>", chunk_size=1024 * 1024):
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            parts = (buffer + chunk).split(separator)

            for document in parts[:-1]:
                if document.strip():
                    yield document

            buffer = parts[-1]

        if buffer.strip():
            yield buffer

def reservoir_sample(documents, k=10, seed=42):
    rng = random.Random(seed)
    reservoir = []

    for i, document in enumerate(documents):
        if i < k:
            reservoir.append(document)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = document

    return reservoir

path_tiny_pkl = "assignment1-basics/data/TinyStoriesV2-GPT4-train.pkl"
path_tiny_txt = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
documents_ts = iter_documents(path_tiny_txt)
sampled_documents_ts = reservoir_sample(documents_ts, k=10, seed=42)
with open(path_tiny_pkl, "rb") as f:
    vocab, merges = pickle.load(f)
    special_tokens = ["<|endoftext|>"]
    tokenizer_ts = Tokenizer(vocab,merges, special_tokens)
comp_ts = compression_ratio(tokenizer_ts, sampled_documents_ts)
print(f"TinyStories (10 documents): {comp_ts:.4f} bytes/token")

path_owt_pkl = "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/owt_train.pkl"
path_owt_txt = "assignment1-basics/data/owt_valid.txt"
documents_owt = iter_documents(path_owt_txt)
sampled_documents_owt = reservoir_sample(documents_owt, k=10, seed=42)
with open(path_owt_pkl, "rb") as f:
    vocab, merges = pickle.load(f)
    special_tokens = ["<|endoftext|>"]
    tokenizer_owt = Tokenizer(vocab,merges, special_tokens)
comp_owt = compression_ratio(tokenizer_owt, sampled_documents_owt)
print(f"OWT (10 documents): {comp_owt:.4f} bytes/token")

comp_owt_with_tstokenizer = compression_ratio(tokenizer_ts, sampled_documents_owt)
print(f"OWT (10 documents) tokenizing with TinyStories tokenizer: {comp_owt_with_tstokenizer:.4f} bytes/token")

comp_ts_with_owttokenizer = compression_ratio(tokenizer_owt, sampled_documents_ts)
print(f"TS (10 documents) tokenizing with OWT tokenizer: {comp_ts_with_owttokenizer:.4f} bytes/token")

# Part C: Throughput is the ratio between (Total UTF-8 BYTES encoded)/(Encoding Time(Seconds))
def measure_throughput(tokenizer, documents):
    total_bytes = 0
    for document in documents:
        num_bytes = len(document.encode("utf-8"))
        total_bytes += num_bytes
    t1 = time.perf_counter()
    for document in documents:
        tokenizer.encode(document)
    t2 = time.perf_counter()
    elapsed = t2 - t1
    if elapsed <= 0:
        raise ValueError("Invalid encoding time")

    throughput = total_bytes / elapsed
    return throughput


throughput_ts = measure_throughput(tokenizer_ts, sampled_documents_ts)
print(f"Throughput of the TS tokenizer in 10 documents of TS is {throughput_ts} bytes/sec")
throughput_owt = measure_throughput(tokenizer_owt, sampled_documents_owt)
print(f"Throughput of the owt tokenizer in 10 documents of owt is {throughput_owt} bytes/sec")
pile_size = 825 * 10**9
time_seconds = pile_size / throughput_owt
time_hours = time_seconds/3600
print(f"Estimated Pile tokenization time: {time_hours:.2f} hours")
