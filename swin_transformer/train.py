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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from swin_transformer import SwinTransformer  # 직접 구현한 모델 사용 시


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
    pass


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
    pass


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

# TODO: main() 함수 구현
# if __name__ == "__main__":
#     main()
