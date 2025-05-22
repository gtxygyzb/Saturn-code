import json
import random
from math import comb
import argparse
import os

def generate_dataset(n_sat, k, result_path, n_sample):
    """
    随机采样生成数据集：
      1. 从 k 个变量中随机选择 n_sat 个变量
      2. 随机生成这 n_sat 个变量的布尔解（solution），总共有 2^n_sat 种可能
      3. 随机对这 n_sat 个变量取反，但必须满足不能全取反（全取反对应 clause 在该 solution 下均为 false）
         合法的取法有 2^n_sat - 1 种
      4. 对未被选中的变量，随机赋值 0/1
      5. 重复采样直到生成 n_sample 个 (solution, clause) 唯一组合
    """
    variables = [chr(ord('A') + i) for i in range(k)]
    samples = {}
    sample_id = 0

    while len(samples) < n_sample:
        # 1. 随机选择 n_sat 个变量
        comb_vars = sorted(random.sample(variables, n_sat))
        # 2. 随机生成这 n_sat 个变量的布尔解（0 或 1）
        sol_bits = [random.choice([0, 1]) for _ in range(n_sat)]
        sol_dict = {var: bit for var, bit in zip(comb_vars, sol_bits)}
        # 3. 随机选择合法的取反方式：在 [1, 2**n_sat - 1] 之间选择一个整数
        pattern = random.randrange(1, 2**n_sat)
        clause_literals = []
        for i, var in enumerate(comb_vars):
            bit = (pattern >> i) & 1
            if bit == 1:
                # literal 与 solution 保持一致：如果对应位为1，则 literal 为变量本身，否则为取反形式
                literal = var if sol_bits[i] == 1 else f'!{var}'
            else:
                # literal 与 solution 相反
                literal = f'!{var}' if sol_bits[i] == 1 else var
            clause_literals.append(literal)
        clause_str = ''.join(clause_literals)

        # 4. 对未选中的变量随机赋值，构造完整的 solution（长度为 k）
        full_solution = []
        for var in variables:
            if var in sol_dict:
                full_solution.append(str(sol_dict[var]))
            else:
                full_solution.append(random.choice(['0', '1']))
        solution_str = ''.join(full_solution)

        # 重复判断：solution 和 clause 相同视为重复
        key = (solution_str, clause_str)
        if key not in samples:
            samples[key] = {
                "id": sample_id,
                "n_sat": n_sat,
                "k": k,
                "solution": solution_str,
                "clause": clause_str
            }
            sample_id += 1

    # 写入文件，每行一个 JSON 对象
    with open(result_path, 'w') as f:
        for sample in samples.values():
            f.write(json.dumps(sample) + "\n")
    
    return len(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a single dataset for SAT problems.")
    parser.add_argument("--n_sat", type=int, required=True, help="Number of variables in a clause.")
    parser.add_argument("--k", type=int, required=True, help="Total number of variables.")
    parser.add_argument("--result_dir", type=str, required=True, help="Result directory.")

    args = parser.parse_args()

    n_sat = args.n_sat
    k = args.k
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    n_sample = min(10000, comb(k, n_sat) * (2**k) * ((2**n_sat) - 1))

    result_path = f"{result_dir}/n{n_sat}_k{k}_single{n_sample}.jsonl"
    num_samples = generate_dataset(n_sat, k, result_path, n_sample)
