"""
=============================================================================
Swin Transformer 학습 가이드
- Phase 1: ImageNet Pre-trained 가중치 로드
- Phase 2: Tiny-ImageNet으로 Fine-tuning
=============================================================================

[학습 전략 개요]
1. Scratch 학습은 GPU 자원이 많이 필요 → Pre-trained 가중치 활용
2. Classification head만 Tiny-ImageNet(200 classes)에 맞게 교체
3. Fine-tuning 진행

[Tiny-ImageNet 데이터셋 정보]
- 200 클래스, 학습 100,000장, 검증 10,000장
- 이미지 크기: 64×64 → 224×224로 resize 필요
- 다운로드: http://cs231n.stanford.edu/tiny-imagenet-200.zip
"""

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.models import swin_t, Swin_T_Weights
import torchvision.transforms as transforms
import time

from knockknock import discord_sender

from swin_transformer import SwinTransformer  # 직접 구현한 모델 사용 시


# =============================================================================
# Step 1: Tiny-ImageNet val 폴더 재구성
# =============================================================================
def reorganize_val_folder():
    """
    Tiny-ImageNet의 val 폴더를 ImageFolder 형식으로 재구성.

    ⚠️ val 폴더는 클래스별 하위 폴더가 없어 ImageFolder로 바로 로드 불가.
       val_annotations.txt를 파싱해서 클래스별로 이미지를 분류해야 함.

    [구현 힌트]
    - val_annotations.txt 각 줄: "파일명\t클래스명\t..." 형식
    - os.makedirs로 클래스 폴더 생성 → shutil.copy2로 이미지 이동
    """
    # TODO: val_annotations.txt 파싱 → 클래스별 폴더 생성 → 이미지 복사
    pass


# =============================================================================
# Step 2: 데이터 Transform + DataLoader
# =============================================================================
"""
[Transform 가이드]
- 학습: Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ColorJitter → Normalize
- 검증: Resize(256) → CenterCrop(224) → Normalize
- 정규화 값: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- (선택) RandAugment, Mixup/CutMix 등 고급 augmentation 적용 가능
"""

# TODO: train_transform, val_transform 정의
# TODO: ImageFolder로 데이터셋 로드
# TODO: DataLoader 구성 (batch_size=64~128, num_workers=4, pin_memory=True)
def set_dateset(batch_size, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader


# =============================================================================
# Step 3: Pre-trained 가중치 로드 + Head 교체
# =============================================================================
"""
[방법 1: timm 라이브러리 - 권장]
- timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
- model.reset_classifier(num_classes=200)

[방법 2: torchvision]
- torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
- model.head = nn.Linear(model.head.in_features, 200)

[방법 3: 직접 구현한 모델]
- SwinTransformer(num_classes=1000) 생성 → 가중치 로드(strict=False) → head 교체
- ⚠️ state_dict 키 이름 매핑이 필요할 수 있음

[Fine-tuning 전략]
- 전략 A: 전체 파라미터 학습 (데이터 충분할 때)
- 전략 B: Head만 학습 (빠른 실험용)
- 전략 C: 점진적 unfreezing (추천) - 처음 head만 → 이후 전체
"""

# TODO: 위 방법 중 하나를 선택하여 모델 로드 + head 교체
# TODO: fine-tuning 전략 선택 후 requires_grad 설정


# =============================================================================
# Step 4: 옵티마이저 & 스케줄러
# =============================================================================
"""
[옵티마이저]
- AdamW 권장 (weight_decay=0.05)
- Fine-tuning lr: 1e-4 ~ 5e-5 (scratch보다 10~100배 작게)
- (팁) backbone과 head에 다른 lr 적용 가능 (param_groups 활용)

[스케줄러]
- CosineAnnealingLR (T_max=총 epoch, eta_min=1e-6)
- (선택) Warmup 추가: timm.scheduler.CosineLRScheduler 또는 직접 구현

[손실 함수]
- CrossEntropyLoss (label_smoothing=0.1 추천)
"""

# TODO: optimizer, scheduler, criterion 정의



# =============================================================================
# Step 5: 학습 루프
# =============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    [구현 힌트]
    1. model.train()
    2. 배치 순회: forward → loss → zero_grad → backward → (gradient clipping) → step
    3. (팁) clip_grad_norm_(max_norm=5.0)으로 학습 안정성 향상
    4. epoch 평균 loss, accuracy 반환
    """
    # TODO: 학습 루프 구현
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 예측 및 손실 계산
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 2. 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # [추가] Gradient Clipping (Swin 같은 Transformer 모델에 권장)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        # 통계 계산
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch % 100 == 0:
            print(f"Batch [{batch}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc



# =============================================================================
# Step 6: 검증 루프
# =============================================================================
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    [구현 힌트]
    1. model.eval() + torch.no_grad()
    2. 배치 순회하며 loss, Top-1 accuracy 계산
    3. (선택) Top-5 accuracy: outputs.topk(5, dim=1) 활용
    """
    # TODO: 검증 루프 구현
    model.eval()  # 평가 모드 (드롭아웃, 배치노름 등이 평가용으로 바뀜)
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 1. 예측 및 손실 계산 (No Grad 상태이므로 backward는 안 함)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # 통계 계산
        total += labels.size(0)
        
        # Top-1 Accuracy 계산
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        
        # Top-5 Accuracy 계산 (예측값 중 상위 5개 안에 정답이 있는지 확인)
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc_top1 = 100. * correct_top1 / total
    val_acc_top5 = 100. * correct_top5 / total
    
    print(f"Validation | Loss: {val_loss:.4f} | Top-1 Acc: {val_acc_top1:.2f}% | Top-5 Acc: {val_acc_top5:.2f}%")
    
    return val_loss, val_acc_top1


# =============================================================================
# Step 7: 메인 학습 실행
# =============================================================================
"""
[전체 흐름]
1. 데이터 준비 (reorganize_val_folder 최초 1회)
2. 모델 준비 (pre-trained 로드 + head 교체)
3. 학습 루프: train → validate → scheduler.step → 로그 출력
4. Best 모델 저장: val_acc 기준으로 체크포인트 저장
   - torch.save({'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc'}, path)
5. 총 30 epoch 정도 권장
"""

def main():
    batch_size = 64
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = set_dateset(batch_size)

    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, 200)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=.1)

    best_acc = 0.0  # 가장 높은 정확도를 저장할 변수
    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")
        print("-" * 20)

        start_time = time.time()
        # 1. 학습 (인자들을 마저 채웠습니다.)
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        # 2. 검증
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {e+1} finished in {epoch_time:.2f} seconds")

        # 3. 스케줄러 스텝 (학습률 갱신)
        scheduler.step()
        
        # 4. Best 모델 저장 (val_acc 기준)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f" Best accuracy 모델 갱신! ({best_acc:.2f}%) 저장 중...")
            
            checkpoint = {
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, 'swin_tiny_best.pth')

# TODO: main() 함수 구현
if __name__ == "__main__":
    main()
