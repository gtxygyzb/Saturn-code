import torch
import re
from typing import Dict, Tuple, Optional
import sys


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     print("[Error] Failed to locate model response header")
    #     return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    processed_str = solution_str
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def reward_func(queries, prompts, labels, format_reward: float = 1.0, answer_reward: float = 1.0):
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
        solution_str = response

        try:
            # Look for the answer pattern in the response
            solution_text = clause
            gt_status = parse_solution_text_format(solution_text)
            expected_names = list(gt_status.keys())
            print(f"[Ground Truth] Final identities: {gt_status}")

            # Extract model answer
            answer_text, processed_str = extract_solution(solution_str)
            print(f"\n[Model Response]\n{processed_str}")

            # Validate response structure
            format_correct = validate_response_structure(processed_str)
            format_score = format_reward if format_correct else -abs(format_reward)
            print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
            print(f"  Format score: {format_score}")

            # Validate answer content
            answer_score = 0
            if format_correct and answer_text:
                pred_status = parse_model_answer(answer_text, expected_names)
                if pred_status:
                    print(f"\n[Content Validation]")
                    print(f"  Expected: {gt_status}")
                    print(f"  Predicted: {pred_status}")
            
                    if pred_status == gt_status:
                        answer_score = 2.0
                        print("  Content validation: FULL MATCH")
                    else:
                        answer_score = -1.5
                        print("  Content validation: MISMATCH")
                else:
                    answer_score = -2.0
                    print( "Fail to parse answer")
            else:
                answer_score = -2.0
                print("\n[Content Validation] Skipped due to format errors or missing answer")

            reward = format_score + answer_score
        except Exception as e:
            reward = -3.0
        print("reward: ", reward)
        reward = float(reward)
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)

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