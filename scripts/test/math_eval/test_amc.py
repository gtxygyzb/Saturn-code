import warnings
warnings.filterwarnings("ignore")
import re
import os
import csv
import json
import time
import types
import random
import textwrap
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from vllm import LLM, SamplingParams

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output")

    args = parser.parse_args()
    model_path = args.model_path
    output_dir = args.output_dir

    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=2,
        max_model_len=32768,
        gpu_memory_utilization=0.6,
    )
    
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
    )

    with open("./scripts/test/math_eval/amc.jsonl", encoding="utf-8") as file:
        data = [json.loads(line) for line in file.readlines() if line]
    
    cnt = 0
    total_time = 0
    results = []

    for d in tqdm(data):
        prompt = d["problem"]
        if "1M" not in model_path:
            messages = [
                {"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=prompt)}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\\boxed{}\\) </answer>."},
                {"role": "user", "content": prompt}
            ]
        
        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        expected_answer = d['answer']
        start_time = time.time()
        outputs = llm.generate([text], sampling_params)
        time_taken = time.time() - start_time
        response = outputs[0].outputs[0].text.strip()

        if '<answer>' in response:
            result = re.split(r'<answer>', response)[1]
        else:
            result = response[len(response) - 30:]
        
        correct = str(int(expected_answer)) in result
        
        result = {
            "question": d['problem'],
            "generated_output": response,
            "expected_expected_answer": expected_answer,
            "correct": correct,
            "time_taken": time_taken
        }

        results.append(result)

        if correct:
            cnt += 1

        total_time += time_taken
    

    acc = cnt / len(data)
    print(f"ACC: {acc}")
    output_metircs = {
        "ACC": acc,
        "total_time": total_time,
        "average_time": total_time / len(data)
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "amc_metrics.json"), 'w') as outfile:
        json.dump(output_metircs, outfile, indent=4)
    
    output_json = os.path.join(output_dir, "amc_output.json")
    with open(output_json, 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()