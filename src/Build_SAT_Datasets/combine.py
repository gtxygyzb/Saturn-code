import json
import random
from math import comb
import argparse

def combine_dataset(single_file, multi_file, n_sat, k, length, n_sample):
    # 读取单子句数据集并按 solution 分组
    solution_groups = {}
    with open(single_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            sol = data["solution"]
            solution_groups.setdefault(sol, []).append(data["clause"])
    
    # 计算总有效组合数，并筛选出满足条件的 solution 组
    total_combinations = 0
    valid_solutions = []
    for sol, clauses in solution_groups.items():
        m = len(clauses)
        if m >= length:
            cnt = comb(m, length)
            total_combinations += cnt
            valid_solutions.append((sol, clauses))
    
    if total_combinations < n_sample:
        n_sample = total_combinations
    
    # 随机采样生成组合，不生成所有可能的组合
    unique_samples = {}
    attempts = 0
    max_attempts = n_sample * 10  # 防止无限循环的尝试次数上限
    while len(unique_samples) < n_sample and attempts < max_attempts:
        attempts += 1
        # 随机选择一个有效的 solution 组
        sol, clauses = random.choice(valid_solutions)
        # 随机选取 length 个子句，顺序无关，所以对选出的组合进行排序以标准化表示
        combo = random.sample(clauses, length)
        sorted_combo = sorted(combo)
        combined_clause = " & ".join(sorted_combo)
        key = (sol, combined_clause)
        if key not in unique_samples:
            unique_samples[key] = {"solution": sol, "clause": combined_clause}
    
    if len(unique_samples) < n_sample:
        return -1  # 表示尝试次数用尽，无法采样足够的唯一样本
    
    # 如果采样得到的样本多于需要的数量，则随机选取 n_sample 个
    selected_samples = random.sample(list(unique_samples.values()), n_sample)
    
    # 构建最终样本数据，添加 id、n_sat、k 等信息
    result = []
    for idx, item in enumerate(selected_samples):
        result.append({
            "id": idx,
            "n_sat": n_sat,
            "k": k,
            "solution": item["solution"],
            "clause": item["clause"]
        })
    
    # 写入文件，每行一个 JSON 对象
    with open(multi_file, 'w') as f:
        for sample in result:
            f.write(json.dumps(sample) + '\n')
    
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a multi dataset for SAT problems.")
    parser.add_argument("--n_sat", type=int, required=True, help="Number of variables in a clause.")
    parser.add_argument("--k", type=int, required=True, help="Total number of variables.")
    parser.add_argument("--length", type=int, required=True, help="Length of clause of multi samples to generate.")
    parser.add_argument("--n_sample", type=int, required=True, help="Number of multi samples to generate.")
    parser.add_argument("--result_dir", type=str, required=True, help="Result directory.")
    args = parser.parse_args()

    n_sat = args.n_sat
    k = args.k
    single_sample = min(10000, comb(k, n_sat) * (2**k) * ((2**n_sat) - 1))
    length = args.length
    n_sample = args.n_sample
    result_dir = args.result_dir

    single_file = f"{result_dir}/n{n_sat}_k{k}_single{single_sample}.jsonl"
    multi_file = f"{result_dir}/n{n_sat}_k{k}_length{length}_sample{n_sample}.jsonl"
    
    flag = combine_dataset(single_file, multi_file, n_sat, k, length, n_sample)
    if flag == False:
        print("Not enough valid combinations")
    elif flag == -1:
        print("Sampling attempts exhausted, not enough unique samples")
    else:
        print(f"Dataset generated: {multi_file} with {n_sample} unique samples")
