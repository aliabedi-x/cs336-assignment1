# Stanford CS336 — Language Modeling from Scratch

This repository contains my personal implementations and experiments while independently studying [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/).

The goal is to understand language models from first principles by implementing their core components, experimenting with different approaches, and exploring performance optimizations.

## Progress

### Assignment 1 — Basics (In Progress)

**BPE Tokenizer Training**
- Implemented byte-level Byte Pair Encoding (BPE) from scratch.
- Added multiprocessing for parallel pre-tokenization.
- Optimized BPE training using incremental pair-frequency updates, inverted indexing, and priority queues.
- Trained and benchmarked tokenizers on TinyStories and OpenWebText subsets.

**Tokenizer**
- Implemented BPE encoding with merge priorities.
- Added support for special tokens.
- Working on decoding, streaming tokenization, and testing.

## Technologies

- Python
- PyTorch
- NumPy
- Hugging Face Datasets
- Python multiprocessing

## Project Status

This is an ongoing learning project. Implementations and documentation will be updated as I progress through the course.

## Acknowledgments

Based on Stanford University's CS336 course and its publicly available assignments.

This is an independent self-study project and is not affiliated with or endorsed by Stanford University.

## License

See the [LICENSE](LICENSE) file for the original repository's license information.
