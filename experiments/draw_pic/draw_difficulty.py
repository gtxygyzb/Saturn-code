import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import math
import json

def compute_difficulty(n, k, l, model_name):
    return math.log2(k) + 2 * math.log2(l) - n + k / n


def process_data(model_data, model_name):
    """
    Transform each record into (D, acc, n_sat, k, L) and sort by D in ascending order
    """
    result = []
    for entry in model_data:
        D = compute_difficulty(entry["n_sat"], entry["k"], entry["L"], model_name)
        result.append((D, entry["acc"], entry["n_sat"], entry["k"], entry["L"]))
    result.sort(key=lambda x: x[0])
    return result

def plot_model_results(model_data, model_name, color, marker='x'):
    """
    Plot a single model's difficulty-accuracy plot
     - scatter plot with specified marker and color
     - add (n_sat, k, L) label next to each point
     - linear regression fit with dashed line and show the formula in legend
     - also fit a quadratic polynomial (as an example curve) with another dashed line and show the formula
    """
    processed = process_data(model_data, model_name)
    difficulties = [d for d, acc, n_sat, k, L in processed]
    accuracies   = [acc for d, acc, n_sat, k, L in processed]
    
    # scatter plot
    plt.scatter(difficulties, accuracies, marker=marker, color=color)
    
    # add labels
    # build `texts` list for adjustText
    texts = []
    for (D, acc, n_sat, k, L) in processed:
        offset = -0.003 if model_name.startswith("DeepSeek") else 0.003
        texts.append(
            plt.text(D, acc + offset, f"({n_sat},{k},{L})", fontsize=8, color=color, ha='left', va='bottom')
        )
    
    adjust_text(
        texts,
        x=difficulties,
        y=accuracies,
        force_points=0.5,
        force_text=0.5,    
        expand_points=(2.5, 2.5),
        expand_text=(1.8, 1.8),
        # arrowprops=dict(arrowstyle='-', lw=0.4, color='gray', alpha=0.4),
        only_move={'points': 'y', 'text': 'xy'},
        precision=0.001,
        lim=200
    )
    
    # linearly regression
    coeffs_linear = np.polyfit(difficulties, accuracies, 1)  # acc = a*D + b
    a_lin, b_lin = coeffs_linear
    x_fit = np.linspace(min(difficulties), max(difficulties), 100)
    y_fit_linear = a_lin * x_fit + b_lin

    # R²
    y_pred = a_lin * np.array(difficulties) + b_lin
    y_true = np.array(accuracies)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"Model: {model_name}, R²: {r_squared:.4f}")

    plt.plot(x_fit, y_fit_linear, '--', color=color,
        label=f'{model_name.replace("-pass@3", "")}: pass@3={a_lin:.2f}·D+{b_lin:.2f}')

sample_num = 100
params = {
    (3, 7, 40),
    (3, 5, 25),
    (3, 5, 20),
    (3, 6, 20),
    (3, 7, 20),
    (4, 7, 40),
    (4, 8, 40),
    (4, 7, 20),
    (6, 7, 40),
    (5, 8, 40)
}
result_dir = "./results/Results/SAT"

model_names = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Saturn-1.5B",
    "Saturn-7B",
]
data = dict()
pass_ks = [1, 3, 5, 7, 10]
for pass_k in pass_ks:
    for model_name in model_names:
        key = f"{model_name}-pass@{pass_k}"
        lis = []
        for param in params:
            n_sat, k, L = param
            task_name = f"n{n_sat}_k{k}_length{L}_sample{sample_num}"
            result_file = f"{result_dir}/{task_name}/{model_name}/evaluation_results.json"
            with open(result_file, 'r') as f:
                f_data = json.load(f)
                acc_key = f"pass@{pass_k}"
                lis.append(
                    {
                        "n_sat": n_sat,
                        "k": k,
                        "L": L,
                        "acc": f_data[acc_key],
                    }
                )
        data[key] = lis

def calc_avg(data):
    """
    calculate the average accuracy of a model
    """
    total = 0
    count = 0
    for entry in data:
        total += entry["acc"]
        count += 1
    print("avg:", total / count if count > 0 else 0)
    return total / count if count > 0 else 0

# calc two models pass@3 avg

avg_pass3_1_5B = calc_avg(data["DeepSeek-R1-Distill-Qwen-1.5B-pass@5"])
avg_saturn_1_5B = calc_avg(data["Saturn-1.5B-pass@5"])
#print("1.5B increased pass@3:", avg_saturn_1_5B - avg_pass3_1_5B)

avg_pass3_7B = calc_avg(data["DeepSeek-R1-Distill-Qwen-7B-pass@5"])
avg_saturn_7B = calc_avg(data["Saturn-7B-pass@5"])
#print("7B increased pass@3:", avg_saturn_7B - avg_pass3_7B)


# draw picture
plt.figure(figsize=(6, 4))

plot_model_results(data["DeepSeek-R1-Distill-Qwen-1.5B-pass@3"], "DeepSeek-R1-Distill-Qwen-1.5B-pass@3", color="#1f77b4", marker='x')
#plot_model_results(data["Saturn-1.5B-pass@3"], "Saturn-1.5B-pass@3", color="#17becf", marker='x')
plot_model_results(data["DeepSeek-R1-Distill-Qwen-7B-pass@3"], "DeepSeek-R1-Distill-Qwen-7B-pass@3", color='#d62728', marker='x')
#plot_model_results(data["Saturn-7B-pass@3"], "Saturn-7B-pass@3", color='#ff7f0e', marker='x')

# plot_model_results(data["DeepSeek-R1-Distill-Qwen-1.5B-pass@1"], "DeepSeek-R1-Distill-Qwen-1.5B-pass@1", color='red', marker='x')
# plot_model_results(data["Saturn-1.5B-pass@1"], "Saturn-1.5B-pass@1", color='purple', marker='x')
# plot_model_results(data["DeepSeek-R1-Distill-Qwen-7B-pass@1"], "DeepSeek-R1-Distill-Qwen-7B-pass@1", color='green', marker='x')
# plot_model_results(data["Saturn-7B-pass@1"], "Saturn-7B-pass@1", color='brown', marker='x')


plt.xlabel('Difficulty (D)', fontsize=10)
plt.ylabel('Pass@3 (%)', fontsize=10)
plt.title('SAT Problem Solving: Difficulty vs. Pass@3', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)


legend = plt.legend(loc='upper right', fontsize=8)


plt.tight_layout()

plt.savefig("./experiments/draw_pic/difficulty_vs_pass3.png") # , format="pdf", bbox_inches="tight")