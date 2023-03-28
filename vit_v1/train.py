# Reference:
# Base MNIST Parameter: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from transformer import VisionTransformer
import config

np.random.seed(45)
torch.manual_seed(45)


if __name__ == '__main__':
    # Loading data
    transform = ToTensor()
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    # train_set = MNIST(root='datasets', train=True, download=True, transform=transform)
    # test_set = MNIST(root='datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size)

    # Defining model and training options
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    device = config.device
    model  = VisionTransformer(config.base_mnist).to(device)
    N_EPOCHS = 10
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()

        train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item()

            test_loss = test_loss/len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct / total * 100:.2f}%")
