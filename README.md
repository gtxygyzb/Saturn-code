# **SATURN**: SAT-based Reinforcement Learning to Unleash Language Model Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2405.12345-b31b1b.svg)](https://arxiv.org/abs/2505.16368)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-gtxygyzb/Saturn--7B-yellow.svg)](https://huggingface.co/gtxygyzb/Saturn-7B)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-gtxygyzb/Saturn--1.5B-yellow.svg)](https://huggingface.co/gtxygyzb/Saturn-1.5B)


We propose **SATURN**, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM reasoning. **SATURN** enables scalable task construction, rule-based verification, and precise difficulty control. **SATURN** designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.

We introduce **SATURN-2.6K**, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply **SATURN** to DeepSeek-R1-Distill-Qwen and obtain **SATURN-1.5B** and **SATURN-7B**.

# 📊 Dataset

Building upon the `SAT_Construction` tool and our difficulty estimation, we release **SATURN-2.6K**, a curated benchmark designed to evaluate LLMs' reasoning capability across varying complexity levels.

**SATURN-2.6K** consists of:
- **1,500 training instances** and **160 test instances** sharing the same estimated difficulty level.
- **1,000 additional test instances** from **10 unseen harder difficulty levels**, with **100 instances per level**.

These difficulty levels are selected based on our estimation function *D(n, k, l)*, enabling a systematic analysis of how LLM performance changes as problem difficulty increases.

The datasets path is:

```bash
./data
```

Additionally, custom datasets with target difficulty levels can be generated using our open-sourced `SAT_Construction` tool (See Step 1 below).

# Models
- **Download:** <https://zenodo.org/records/15487151>
- **Hugging Face:** [Saturn-7B](https://huggingface.co/gtxygyzb/Saturn-7B), [Saturn-1.5B](https://huggingface.co/gtxygyzb/Saturn-1.5B)

# 🧱 Installation

To install the required dependencies, run:

```bash
conda create -n saturn python=3.10.12
conda activate saturn
pip install -r requirements.txt
```

# 🛠️ Usage Guide

## 1. SAT Data Construction

Run the following script:

```bash
sh ./src/Build_SAT_Datasets/build_sat_dataset.sh
````

Edit the following variables in the script to configure difficulty and number of samples:

```bash
PARAMETERS=( 
  "3 5 20" 
) 
N_SAMPLE=520
```

This controls the SAT problem's (n, k, l) parameters and sample count.

## 🚀 2. **SATURN** Model Training

Training scripts are located in:

```
scripts/train
```

We provide separate scripts for both the **1.5B** and **7B** models. Each stage of training is isolated for better observability and debugging. For example:
```
sh ./scripts/train/grpo_1.5B_355.sh
```

### 🔧 Required Arguments

Before running the script, please modify the following parameters:

```bash
--pretrain /xxx/Qwen \
--save_path xxx \
--use_wandb xxx \
--wandb_run_name xxx \
--ckpt_path xxx/checkpoints \
```

### 📚 Full Argument List

For more detailed argument configurations, please refer to the [OpenRLHF documentation](https://github.com/OpenRLHF/OpenRLHF).


## 3. **SATURN** Benchmark Evaluation

Run:

```bash
sh ./scripts/test/test_SAT.sh
```

Edit the first two lines in the script before running:

```bash
model_path= # TODO: your local model path
model_name= # TODO: name you want to assign
```

We use Docker + vLLM to deploy models. You should modify Docker parameters like `-v` based on your server setup. You may also modify vLLM-related arguments in the script. See [vLLM](https://github.com/vllm-project/vllm) for reference.


## 4. Math and Programming Benchmark Evaluation

Run:

```bash
sh ./scripts/test/test_model_math_programming.sh
```

Modify the third line:

```bash
MODEL= # TODO: model path
```

Other arguments follow [lighteval](https://github.com/huggingface/lighteval) conventions.


## 5. Experimental Results

### 5.1 Word Frequency and Word Cloud

To generate a word cloud, uncomment line 39 in:

```bash
sh ./scripts/test/test_model.sh
```

```bash
#python ./scripts/test/frequency_cloud.py \
#  --work_dir $OUTPUT_DIR \
#  --model $MODEL
```

### 5.2 SAT Difficulty Estimation

To reproduction Figure 3, run:

```bash
python ./experiments/draw_pic/draw_difficulty.py
```

Figures will be saved in `./experiments/draw_pic/`.


# 🤝 Acknowledgements

This project reuses code from the following repositories:

* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
* [vLLM](https://github.com/vllm-project/vllm)
* [lighteval](https://github.com/huggingface/lighteval)
* [open-r1](https://github.com/huggingface/open-r1)

# 📜 Citation

```

```

# 📄 License

This repository includes components licensed under the Apache License 2.0.
