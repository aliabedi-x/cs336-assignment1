import numpy as np
import os
from cs336_basics.pretokenization_example import Tokenizer


paths = [
    "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/TinyStoriesV2-GPT4-valid-fixed.bin",
    "data/TinyStoriesV2-GPT4-train.bin",
    "data/owt_valid.bin"
]

for path in paths:
    tokens = np.memmap(path, dtype=np.uint16, mode="r")

    print(f"\n{path}")
    print(f"Shape: {tokens.shape}")
    print(f"dtype: {tokens.dtype}")
    print(f"Total tokens: {tokens.size}")
    print(f"First 20 token IDs: {tokens[:20]}")
    print(f"Minimum token ID: {tokens.min()}")
    print(f"Maximum token ID: {tokens.max()}")
    print(f"File size (bytes): {os.path.getsize(path)}")


# Load TinyStories tokenizer
tokenizer = Tokenizer.from_files(
    "data/TinyStoriesV2-GPT4-train.pkl",
    special_tokens=["<|endoftext|>"]
)


# Memory mapping
train_data = np.memmap(
    "data/TinyStoriesV2-GPT4-train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    "/Users/hooshbang/Desktop/assignment1/assignment1-basics/data/TinyStoriesV2-GPT4-valid-fixed.bin",
    dtype=np.uint16,
    mode="r"
)


# Decoding verification
print("\n--- Training Decoding Verification ---")
print(tokenizer.decode(train_data[:100].tolist()))

print("\n--- Validation Decoding Verification ---")
print(tokenizer.decode(val_data[:100].tolist()))


# End-of-text verification
eot_id = tokenizer.ids[b"<|endoftext|>"]

print("\n--- EOT Verification ---")
print("EOT ID:", eot_id)


# Memory-efficient EOT counting
def count_eot(data, eot_id, chunk_size=1_000_000):
    total = 0

    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size]
        total += np.count_nonzero(chunk == eot_id)

    return total


print("Training EOT count:", count_eot(train_data, eot_id))
print("Validation EOT count:", count_eot(val_data, eot_id))