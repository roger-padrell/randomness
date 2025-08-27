import torch
import os
from pathlib import Path
from alive_progress import alive_bar

CTX_LEN = 128
PROPORTION_PRECISION = 5
script_dir = Path(__file__).resolve().parent

def fix_length(nums, target_length=CTX_LEN):
    if len(nums) < target_length:
        return [0] * (target_length - len(nums)) + nums
    elif len(nums) > target_length:
        return nums[-target_length:]
    else:
        return nums

def digit_proportions(digits):
    total = len(digits)
    if total == 0:
        return [0.0] * 10
    proportions = [0.0] * 10
    for d in digits:
        proportions[d] += 1
    proportions = [round(count / total, PROPORTION_PRECISION) for count in proportions]
    return proportions

def parse_num(nstr):
    inps, oups = [], []
    for i in range(len(nstr) - 1):
        l = list(map(int, list(nstr[:i+1])))
        p = digit_proportions(l)
        inps.append(fix_length(l) + p)
        oups.append(int(nstr[i+1]))   # store as class index (0–9)
    return inps, oups

def parse_folder(path):
    inps, oups = [], []
    files = [f for f in Path(path).iterdir() if f.is_file()]
    with alive_bar(len(files)) as bar:
        for entry in files:
            with open(entry, "r", encoding="utf-8") as file:
                for line in file:
                    X, Y = parse_num(line.strip())
                    inps.extend(X)
                    oups.extend(Y)
            bar()
    return inps, oups

if __name__ == "__main__":
    print("Preprocessing dataset...")
    X_data, Y_data = parse_folder(script_dir / "../../DATA/rng")

    # Convert to tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.long)  # long for class indices

    print("Saving preprocessed tensors...")
    torch.save((X_tensor, Y_tensor), script_dir / "../../DATA/rng.pt")
    print("Done! Saved as rng.pt")

