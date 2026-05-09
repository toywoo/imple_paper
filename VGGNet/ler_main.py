import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from dataset import get_dataloaders
from ler_model import VGGNet_D_BN
from ler_train import train_one_epoch
from ler_test import test
import matplotlib.pyplot as plt
from knockknock import discord_sender

def main():
    SET_T = "ler_D_multi_BN"
    epoch = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로더 (Multi 모드)
    train_data, test_data = get_dataloaders(isSingle=False)
    
    # 모델 설정 (VGG-D)
    model = VGGNet_D_BN().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=5e-4, lr=1e-2, momentum=.9)
    
    # 스케줄러 설정: 30, 50, 70 에폭에서 학습률 1/10 감소
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)

    loss_history = []
    best_acc = 0.0

    for t in range(epoch):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        # 1. 학습 진행
        avg_loss = train_one_epoch(device, train_data, model, loss_fn, optimizer, t+1, loss_history)
        
        # 2. 검증 진행
        if (t + 1) % 5 == 0:
            current_acc = test(device, test_data, model, loss_fn)
        
            # 4. 베스트 모델 저장 (테스트를 했을 때만 체크)
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(model.state_dict(), f"VGG_{SET_T}_best.pth")
                print(f"==> New Best Accuracy: {best_acc:.2f}%! Model Saved.")
        
        # 3. 스케줄러 업데이트
        scheduler.step()

    # 최종 결과 저장
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(f"Training Loss - {SET_T}")
    plt.savefig(f"loss_plot_{SET_T}.png")

    torch.save(model.state_dict(), f"VGG_{SET_T}_final.pth")
    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
