from cs336_basics.pretokenization_example import (
    Tokenizer,
    tokenize_to_binary
)

tokenizer = Tokenizer.from_files(
    "data/TinyStoriesV2-GPT4-train.pkl",
    special_tokens=["<|endoftext|>"]
)

tokenize_to_binary(
    tokenizer,
    input_path="data/TinyStoriesV2-GPT4-valid.txt",
    output_path="data/TinyStoriesV2-GPT4-valid-fixed.bin"
)

print("Validation tokenization completed.")