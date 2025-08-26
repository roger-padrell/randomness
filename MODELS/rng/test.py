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
from alive_progress import alive_bar
import glob
import os
import argparse

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("-s", action="store_true", help="Silent, hide logs")
parser.add_argument("-a", action="store_true", help="Advanced mode, more info")
parser.add_argument("-r", type=int, default=10, help="Total runs (default 10)")
parser.add_argument("-m", type=str, required=False, help="Model path")
args = parser.parse_args()

# SETUP
silent = args.s;
adv = args.a;
TOTAL_RUNS = args.r;

if args.m is None:
    pth_files = glob.glob(os.path.join(script_dir, "*.pth"))
    for f in range(len(pth_files)):
        print(f"{f}: {os.path.basename(pth_files[f])}")
    
    mname = pth_files[int(input("Select model (by number): "))];
else:
    mname = args.m;
if not silent:
    print(f"Selected: {mname}");
    print("Loading model...")

model = torch.load(mname, map_location="cpu", weights_only=False)
def parse_out(out):
    props = out.tolist();
    return props[0].index(max(props[0]))
    
def parse_out_adv(out):
    props = out.tolist();
    return [props[0].index(max(props[0])), props[0]];
    
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
    
def single_run():
    n = str(random_number());
    total = 0;
    correct = 0;
    for i in range(len(n)):
        if i == len(n)-1:
            break;
        predicted = parse_out(evaluate(n[:i]));
        actual = n[i+1];
        total += 1;
        if int(predicted) == int(actual):
            correct += 1;
    return (correct, total);
    
def single_run_adv():
    n = str(random_number())
    total = 0
    correct = 0
    props = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(n)):
        if i == len(n)-1:
            break
        predicted = parse_out_adv(evaluate(n[:i]))
        
        for pr in range(len(predicted[1])):
            props[pr] += predicted[1][pr]

        actual = n[i+1]
        total += 1
        if int(predicted[0]) == int(actual):
            correct += 1
    return [correct, total, props]

if __name__ == "__main__":
    total_n = 0
    correct = 0
    props = [0,0,0,0,0,0,0,0,0,0]
    if silent:
        pout = sys.stderr;
    else:
        pout = sys.stdout;
        print("Starting evaluation...")
    with alive_bar(TOTAL_RUNS, file=pout) as bar:
        for i in range(TOTAL_RUNS):
            if adv is True:
                r = single_run_adv()
                for pr in range(len(r[2])):
                    props[pr] += r[2][pr]
            else:
                r = single_run()

            total_n += r[1]
            correct += r[0]
            bar()
	
    print(f"Correct: {correct}/{total_n} ({round((correct/total_n)*100, 2)}%)")

    if adv:
        print("\nProportions:")
        for pr in range(len(props)):
            props[pr] = props[pr] / total_n
            print(f"\t{pr}: {round(props[pr]*100, 2)}%")

