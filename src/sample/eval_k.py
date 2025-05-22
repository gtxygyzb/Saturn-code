import re
import os
import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def calc_sat_value(clause, solution):
    def parse_literals(clause_str):
        literals = []
        i = 0
        while i < len(clause_str):
            if clause_str[i] == '!':
                literals.append(clause_str[i:i+2])
                i += 2
            else:
                literals.append(clause_str[i])
                i += 1
        return literals
    
    for subclause in clause.split(' & '):
        satisfied = False
        for lit in parse_literals(subclause):
            neg = False
            if lit.startswith('!'):
                var = lit[1]
                neg = True
            else:
                var = lit
            
            idx = ord(var) - ord('A')
            if idx >= len(solution):
                val = '0'
            else:
                val = solution[idx]
            
            if (neg and val == '0') or (not neg and val == '1'):
                satisfied = True
                break
        
        if not satisfied:
            return 0
    return 1


parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")
args = parser.parse_args()

work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path

save_dir = os.path.join(work_dir, model_name)
result_file = os.path.join(save_dir, "results.jsonl")
failed_file = os.path.join(save_dir, "failed_examples.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_path)

def truncate(text: str) -> str:
    matches = re.findall(r'\\boxed{(.*?)}', text)
    return matches[-1] if matches else None

length = 0 
total_tokens = 0 
pass_k = [1, 3, 5, 7, 10] 
pass_counts = {k: 0 for k in pass_k}

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


with open(result_file, "r") as f:
    for line in tqdm(f):
        length += 1
        example = json.loads(line)
        clause = example["clause"]
        completions = example["completions"]  # completions is a list
        
        # total tokens numbers
        total_tokens += sum(len(tokenizer.tokenize(comp)) for comp in completions) / len(completions)


        # truncate completions，filter None
        solutions = [truncate(comp) for comp in completions]
        valid_solutions = [sol for sol in solutions if sol is not None]

        # calculate correct solutions
        c = sum(1 for sol in valid_solutions if calc_sat_value(clause, sol))
        n = len(solutions)

        # calulate pass@k
        for k in pass_k:
            pass_counts[k] += pass_at_k(n, c, k)

# final results
pass_rates = {f"pass@{k}": pass_counts[k] / length for k in pass_k}
avg_tokens = total_tokens / length

evaluation_results = {
    "model_name": model_name,
    **pass_rates,
    "average_tokens": avg_tokens,
}

output_file = os.path.join(save_dir, "evaluation_results.json")
os.makedirs(save_dir, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Results saved to {output_file}")
