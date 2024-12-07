"""
Resources
---------
https://stackoverflow.com/a/68609343/8728749
https://pytorch.org/docs/stable/optim.html
https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
https://github.com/AGnias47/brats-challenge-cis-5528/blob/main/nn/nnet.py
https://pytorch.org/torcheval/main/generated/torcheval.metrics.R2Score.html
https://piexchange.medium.com/decoding-deep-learning-neural-networks-for-regression-part-i-332f1d2fedd5#:~:text=The%20number%20of%20neurons%20in,considered%20a%20deep%20neural%20network.
HW 5/6
"""

import sys
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F
from rainbow_tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import MeanSquaredError, R2Score

sys.path.append(".")
from models.constants import PRECISION, RANDOM_STATE
from models.imdb_dataloader import train_test_val

BATCH_SIZE = 64
EPOCHS = 25

torch.manual_seed(RANDOM_STATE)


class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self):
        super(FeedforwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=62, out_features=45)
        self.fc2 = nn.Linear(in_features=45, out_features=23)
        self.fc3 = nn.Linear(in_features=23, out_features=12)
        self.fc4 = nn.Linear(in_features=12, out_features=1)
        self.alpha = 0.01
        self.gamma = 0.9
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def save(self, filename=None):
        if not filename:
            filename = f"{str(uuid4())}.pth"
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


def train(model, device, dataloader, epochs=10):
    model.train()
    optimizer = Adam(model.parameters(), lr=model.alpha)
    scheduler = ExponentialLR(optimizer, gamma=model.gamma)
    total_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            with torch.set_grad_enabled(True):
                prediction = model(X).flatten()
                loss = model.loss_function(prediction, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Loss: {round(epoch_loss, PRECISION)}")
        total_loss.append(epoch_loss)
    return min(total_loss)


def test(model, device, dataloader):
    model.eval()
    mse = MeanSquaredError(device=device)
    r2_score = R2Score(device=device)
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            prediction = model(X).flatten()
            mse.update(prediction, Y)
            r2_score.update(prediction, Y)
    return mse.compute(), r2_score.compute()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not detected; not running Neural Net training without a GPU configured"
        )
    device = torch.device("cuda")
    model = FeedforwardNeuralNetwork().to(device)
    train_dataloader, test_dataloader, validation_dataloader = train_test_val(
        batch_size=BATCH_SIZE
    )
    training_loss = train(model, device, train_dataloader, epochs=EPOCHS)
    print(f"Lowest training loss: {training_loss}")
    mse, r2 = test(model, device, validation_dataloader)
    print(f"MSE: {mse}, R2: {r2}")
