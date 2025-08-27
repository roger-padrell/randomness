import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from alive_progress import alive_bar
from pathlib import Path
from datetime import datetime

today = datetime.today()
fdate = today.strftime("%d/%m/%y")

script_dir = Path(__file__).resolve().parent

# ---------------------- Model ----------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(138, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading preprocessed dataset...")
    dataname = input('What dataset to use (rng, human, crypto, ai): ');
    X_tensor, Y_tensor = torch.load(script_dir / f"../DATA/{dataname}.pt")

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	
    if input("Use existing model as base (y/n): ") == "y":
        mpath = input("Existing model path: ")
        model = torch.load(mpath, map_location="cpu", weights_only=False)
        mname = mpath.split(".")[0];
    else:
    	model = Net().to(device)
    	mname = f"{dataname}({fdate})"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("Started training")
    epochs = 20

    for epoch in range(epochs):
        running_loss = 0.0
        with alive_bar(len(dataloader)) as bar:
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                bar()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        torch.save(model, f"{mname}_e{epoch+1}.pth")

