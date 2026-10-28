# Re-evaluation of the DeepSeek-R1-Distill-Qwen-7B AIME baseline at 32k

## 1. Setup

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (downloaded to `/root/Saturn-code/models/`)
- Protocol: a single unified one — lighteval 0.13.0 for **both** AIME24 and AIME25
  (the old mixed lighteval + Logic-RL `test_aime.py` path is not used)
- Sampling: `pass@1`, `temperature=0.6`, `top_p=0.95`, `dtype=bfloat16`
- Hardware: 8x A100-40GB; the two tasks run concurrently
- Script: `scripts/test/fix_bug.sh` (`MAX_LEN=16384 bash scripts/test/fix_bug.sh` for the control)
- Env: `/root/Saturn-code/.venv-eval` (python 3.11, lighteval 0.13.0, vllm 0.10.1.1,
  torch 2.7.1, transformers 4.55.4)

## 2. New results

- AIME24, 16,384 tokens: 46.67 (6/30 truncated, mean 9,252 gen tokens)
- AIME24, 32,768 tokens: 43.33 (1/30 truncated, mean 11,947 gen tokens)
- AIME25, 16,384 tokens: 40.00 (10/30 truncated, mean 10,414 gen tokens)
- AIME25, 32,768 tokens: 43.33 (0/30 truncated, mean 11,668 gen tokens)
- AIME24+25 average: 43.33 at 16k vs 43.33 at 32k

Reference numbers from the paper's released logs (old mixed protocol, 16,384 limit):
AIME24 56.67, AIME25 43.33.

Reading: lifting the cap to 32,768 removes almost all truncation (16/60 -> 1/60 across the
two sets) but does **not** raise the 7B baseline's score — the average over AIME24+25 is
identical (43.33 both ways), and the two sets move in opposite directions.

Raw outputs: `DeepSeek-R1-Distill-Qwen-7B_32768/` and `DeepSeek-R1-Distill-Qwen-7B_16384/`
(lighteval `results/` + `details/` parquet, per-task log, `truncation_audit.{json,md}`).

## 3. Caveats

- lighteval 0.13 is not byte-identical to the March 2025 version (prompt rendering and
  answer extraction changed), so the new numbers are internally comparable (16k vs 32k)
  but only indicative when compared to the published table.
