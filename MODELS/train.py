import torch
import torch.nn as nn
import os
from alive_progress import alive_bar

CTX_LEN = 128;
PROPORTION_PRECISION = 5;

def fix_length(nums, target_length=CTX_LEN):
    if len(nums) < target_length:
        # prepend zeros
        return [0] * (target_length - len(nums)) + nums
    elif len(nums) > target_length:
        # trim from the start
        return nums[-target_length:]
    else:
        # already the right length
        return nums
        
def digit_proportions(digits):
    total = len(digits)
    if total == 0:
        return [0.0] * 10  # avoid division by zero
    
    proportions = [0.0] * 10
    for d in digits:
        proportions[d] += 1
    
    # normalize to proportions
    proportions = [round(count / total, PROPORTION_PRECISION) for count in proportions]
    return proportions

def parse_num(nstr):
    inps = [];
    oups = [];
    for i in range(len(nstr)):
        if i == len(nstr)-1:
            break;
        # Add context param
        l = list(map(int, list(nstr[:i+1])));
        p = digit_proportions(l);   
        inps.append(fix_length(l) + p)
        oups.append(int(nstr[i+1]))
    return (inps, oups);

def parse_file(path):
    inps = []
    oups = []
    with open(path, "r", encoding="utf-8") as file:
        file.seek(0)
        for line in file:
            ns = parse_num(line.strip());
            inps += ns[0];
            oups += ns[1];
    return (inps, oups);

def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

def parse_set(path):
    inps = []
    oups = []
    files = count_files(path);
    with alive_bar(files) as bar:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    f = parse_file(entry.path);
                    inps += f[0];
                    oups += f[1];
                    bar()
    return (inps, oups)
parse_set("../DATA/rng/");

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(138, 256)   # first hidden layer
        self.fc2 = nn.Linear(256, 128)   # second hidden layer
        self.fc3 = nn.Linear(128, 10)    # output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # outputs between 0 and 1
        return x

# X_data = [[...138 numbers...], [...], ...]
# Y_data = [[...10 numbers...], [...], ...]
"""
# Convert to tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

# Create dataset & dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----- Training setup -----
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- Training loop -----
epochs = 20
for epoch in range(epochs):
    for batch_X, batch_Y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
"""