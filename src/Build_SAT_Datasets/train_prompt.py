import json
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a prompt for SAT problems.")
parser.add_argument("--n_sat", type=int, required=True, help="Number of variables in a clause.")
parser.add_argument("--k", type=int, required=True, help="Total number of variables.")
parser.add_argument("--length", type=int, required=True, help="Length of clause of multi samples.")
parser.add_argument("--n_sample", type=int, required=True, help="Number of multi samples.")
parser.add_argument("--start_id", type=int, default=0, help="Start id of the samples.")
parser.add_argument("--seed_datasets_dir", type=str, help="Directory of the seed datasets.")
parser.add_argument("--prompt_dir", type=str, help="Directory to save the generated prompt.")

args = parser.parse_args()

n_sat = args.n_sat
k = args.k
length = args.length
n_sample = args.n_sample
start_id = args.start_id
seed_datasets_dir = args.seed_datasets_dir

seed_datasets_file = os.path.join(seed_datasets_dir, f"n{n_sat}_k{k}_length{length}_sample{n_sample}.jsonl")
prompt_dir = args.prompt_dir

if not os.path.exists(prompt_dir):
    os.makedirs(prompt_dir)

prompt_file = os.path.join(prompt_dir, "train.jsonl")

meta_prompt = """You are now required to solve a SAT (Boolean Satisfiability) problem. The problem is provided in JSON format, containing the following fields:

- "n_sat": The number of variables in each clause (n-SAT problem).
- "k": The total number of distinct variables in the problem.
- "clause": A string representation of the SAT formula, where clauses are separated by " & " (representing logical AND). Within each clause, variables are combined using concatenation (representing logical OR). A negation is indicated by "!" before a variable.

Your task is to provide a valid solution. The answer is a string of length k representing the truth values of the variables in order (1 for true, 0 for false). If there are multiple solutions, provide any one of them.
Please reason step by step, and put your final answer within \\boxed{}.

**Example**
{"n_sat": 3, "k": 4, "clause": "!B!C!D & A!B!D & AB!D"}
**Final Answer**
\\boxed{1101}

Below is the SAT problem you need to solve:
"""

with open(seed_datasets_file, 'r') as f:
    data = f.readlines()
    for sample in data:
        sample = json.loads(sample)
        n_sat = sample["n_sat"]
        k = sample["k"]
        clause = sample["clause"]
        question = {"n_sat": n_sat, "k": k, "clause": clause}
        # question to string
        question = json.dumps(question)
        sample["prompt"] = meta_prompt + question + "\n"
        sample["id"] = start_id
        start_id += 1
        
        if sample["id"] < n_sample-20:
            with open(prompt_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')
        else:
            test_file = os.path.join(prompt_dir, "test.jsonl")
            with open(test_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')

print(f"Prompt file generated successfully: {prompt_file}")
