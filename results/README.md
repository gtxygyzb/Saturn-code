# README

This folder contains the primary experimental results for our submission. We include evaluations on two main tasks:

1. **SATURN-2.6K**: A benchmark for evaluating model performance on SAT tasks.
2. **Math and Programming Benchmark**: AIME 24/25, AMC 22/23, Math500, GPQA-D, and LiveCodeBench.

## Directory Structure

- `Results/SAT/`: Results on SATURN-2.6K.
  - Structured by configuration (e.g., `n3_k6_length20_sample100`)
  - Each folder contains:
    - `evaluation_results.json`: summary metrics
    - `results.jsonl.txt`: model predictions' time (s)
    - `results.json` full model's output

- `Results/Math_and_Programming/`: Results on AIME, AMC, GPQA-D, and LiveCodeBench.
  - Structured by model
  - Each folder contains:
    - `results/`: per-task summaries
    - `details/`: detailed logs and model's outputs in `.parquet` format

## Notes

Due to the 100MB size limit, we have removed some lengthy model output traces. All key evaluation metrics are retained.

For complete results and outputs, please refer to the supplementary repository linked in the paper.
