import torch
import re
import sys

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


def reward_func(queries, prompts, labels):
    """
    Calculate rewards based on how well the model's answers match expected labels.
    
    Args:
        queries (list): Full sequences including prompts and responses
        prompts (list): The input prompts
        labels (list): Expected answers/ground truth (clause)
        
    Returns:
        torch.Tensor: Reward scores for each query
    """
    rewards = []
    
    for query, prompt, clause in zip(queries, prompts, labels):
        # Extract the model's answer from the query
        response = query[len(prompt):]  # Get just the response part
        try:
            # Look for the answer pattern in the response
            matches = re.findall(r'\\boxed{(.*?)}', response)

            predicted_answer = matches[-1] if matches else None
            if predicted_answer is not None:
                if calc_sat_value(clause, predicted_answer):
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = -1.0  # No reward if the answer pattern isn't found
        except Exception as e:
            print(f"error: {e}")
            print(f"Label: {label}")
            print(f"Response: {response[-100:]}...")  # debugging
            reward = -2.0
        
        rewards.append(reward)
    
    return torch.tensor(rewards)

def compute_similarity(text1, text2):
    """
    Compute the similarity between two text strings.
    This is a simple example - you might want to use more sophisticated metrics.
    """
    # Simple Jaccard similarity on word sets
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union