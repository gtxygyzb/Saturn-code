"""
Evaluate whether a given binary solution satisfies a SAT clause in string format.

This script reads a SAT instance from a JSONL file, extracts the clause and solution,
and checks satisfiability using `calc_sat_value`. Each clause is a conjunction of 
disjunctions (CNF-like), and variables are represented as uppercase letters ('A', 'B', ...).

Example use case:
- Verifying model-generated solutions to synthetic SAT prompts.
"""


import json
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
            # analysis literals
            neg = False
            if lit.startswith('!'):
                var = lit[1]
                neg = True
            else:
                var = lit
            
            # get corresponding SAT value
            idx = ord(var) - ord('A')
            if idx >= len(solution):
                val = '0'
            else:
                val = solution[idx]
            
            # check satisfied
            if (neg and val == '0') or (not neg and val == '1'):
                satisfied = True
                break
        
        if not satisfied:
            return 0
    return 1

idx = 0

with open("./data/test/n4_k7_length40_sample100.jsonl", "r") as f:
    lines = f.readlines()
    sample = lines[idx]
    data = json.loads(sample)
    
    print(data["prompt"])
    clause = data["clause"]
    print("solution:", data["solution"])
    solution = data["solution"]
    print(calc_sat_value(clause, solution))
