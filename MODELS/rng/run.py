import torch
import sys
import os
from pathlib import Path
import numpy as np
script_dir = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(script_dir / ".."))
sys.path.append(os.path.abspath(script_dir / "../../GENERATORS/rng"))
from train import Net, fix_length, digit_proportions
from generate import random_number

model = torch.load(script_dir / "medium1.pth", map_location="cpu", weights_only=False)
def parse_out(out):
    props = out.tolist();
    return props[0].index(max(props[0]))
    
def evaluate(number):
    nstr = str(number);
    nums = list(map(int, list(nstr)));
    prop = digit_proportions(nums);
    fixed = fix_length(nums);
    data = np.array(fixed + prop) 
    
    # Convert to torch tensor
    x = torch.tensor(data, dtype=torch.float32)
    
    # Add batch dimension (so shape is [1, input_size])
    x = x.unsqueeze(0)
    
    # Run through the model
    with torch.no_grad():
        output = model(x)
    
    return output;
    
if __name__ == "__main__":
    n = str(random_number());
    correct = 0;
    for i in range(len(n)):
        if i == len(n)-1:
            break;
        predicted = parse_out(evaluate(n[:i]));
        actual = n[i+1];
        if int(predicted) == int(actual):
            correct += 1;
            print(f"{actual} | {predicted} - correct")
            continue;
        print(f"{actual} | {predicted}")
    print("Correct guesses: " + str(correct))
    print("Proportion: " + str(correct/127))