import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from pathlib import Path
from alive_progress import alive_bar
import time

CTX_LEN = 128
PROPORTION_PRECISION = 5
script_dir = Path(__file__).resolve().parent

# ---------------------- Utility functions ----------------------
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
        someout = [0] * 10
        someout[int(nstr[i+1])] = 1
        oups.append(someout)
    return inps, oups

# ---------------------- Custom Dataset ----------------------
class RNGDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.files = [f for f in self.folder.iterdir() if f.is_file()]
        self.index_map = []

        print("Indexing dataset...")
        with alive_bar(len(self.files)) as bar:
            for f_idx, f in enumerate(self.files):
                with open(f, "r", encoding="utf-8") as file:
                    for line_idx, line in enumerate(file):
                        nstr = line.strip()
                        for pos_idx in range(len(nstr) - 1):
                            self.index_map.append((f_idx, line_idx, pos_idx))
                bar()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        f_idx, line_idx, pos_idx = self.index_map[idx]
        filepath = self.files[f_idx]
        with open(filepath, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i == line_idx:
                    nstr = line.strip()
                    break
        l = list(map(int, list(nstr[:pos_idx+1])))
        p = digit_proportions(l)
        inp = fix_length(l) + p
        out = [0] * 10
        out[int(nstr[pos_idx+1])] = 1
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(out, dtype=torch.float32)

# ---------------------- Model ----------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(138, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ---------------------- Main ----------------------
if __name__ == "__main__":
    start = int(time.time() * 1000);

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    print("Using device:", device);
	
    print("Loading dataset lazily...")
    dts = input("Which dataset to use (rng, ai, human, crypto): ")
    dataset = RNGDataset(script_dir / "../DATA/" / dts)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Started training")
    epochs = 20

    for epoch in range(epochs):
        running_loss = 0.0
        with alive_bar(len(dataloader)) as bar:
            for batch_X, batch_Y in dataloader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                bar()
        torch.save(model, script_dir / f"{dts}/{start}_{epotch}.pth")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

