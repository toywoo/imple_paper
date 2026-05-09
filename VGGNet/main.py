import torch
from torch import nn

from dataset import get_dataloaders
from model import *
from train import train
from test import test
import matplotlib.pyplot as plt


def main():
    SET_T = "D_multi_e74"

    train_data, test_data = get_dataloaders(isSingle=False)
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

    epoch = 1
    batch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGGNet_D()
    print(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5e-4, lr=1e-2, momentum=.9)

    loss_history = train(device, train_data, epoch, model, loss_fn, optimizer)

    # Loss 그래프 생성 및 저장
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"loss_plot_{SET_T}.png")
    print("Saved Loss Plot as loss_plot.png")

    torch.save(model.state_dict(), f"VGG_{SET_T}.pth")
    print("Saved PyTorchModel State")

    # 최종 테스트
    test(device, test_data, model, loss_fn)


if __name__ == "__main__":
    main()
