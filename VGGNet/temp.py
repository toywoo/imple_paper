import torch
from torch import nn

from dataset import get_dataloaders
from model import *
from train import train
from test import test
import matplotlib.pyplot as plt

def temp_to_test():
    train_data, test_data = get_dataloaders(isSingle=True)
    print("--- Train Data ---")
    for X, y in train_data:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    print("\n--- Test Data ---")
    for X, y in test_data:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # epoch = 74
    epoch = 5
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGGNet_D()
    print(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5e-4, lr=1e-2, momentum=.9)

    model.load_state_dict(torch.load("VGG_D_multi_e30.pth"))
    test(device, test_data, model, loss_fn)


temp_to_test()
