"""
=============================================================================
Swin Transformer 테스트 & 추론 가이드
- 학습된 모델 로드
- 테스트 데이터셋 평가
- 단일 이미지 추론
- 결과 시각화
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from swin_transformer import SwinTransformer


# =============================================================================
# Step 1: 학습된 모델 로드
# =============================================================================
"""
[방법 1: 직접 구현 모델]
- SwinTransformer(num_classes=200) 생성 → torch.load → load_state_dict

[방법 2: timm 모델]
- timm.create_model(..., num_classes=200) → load_state_dict

⚠️ 체크포인트 dict 구조: {'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc'}
"""

# TODO: 체크포인트 로드 → 모델에 가중치 적용 → model.eval()


# =============================================================================
# Step 2: 테스트 데이터셋 평가
# =============================================================================
@torch.no_grad()
def test(model, test_loader, device):
    """
    [구현 힌트]
    1. model.eval()
    2. Top-1 accuracy: outputs.max(1) → predicted.eq(labels) 비교
    3. Top-5 accuracy: outputs.topk(5, dim=1) 활용
    4. (선택) 클래스별 정확도 추적: dict로 클래스별 correct/total 집계
    5. 가장 잘 맞추는/못 맞추는 클래스 분석하면 모델 이해에 도움
    """
    # TODO: Top-1, Top-5 accuracy 계산
    # TODO: (선택) 클래스별 정확도 분석
    pass


# =============================================================================
# Step 3: 단일 이미지 추론
# =============================================================================
def predict_single_image(model, image_path, transform, class_names, device, top_k=5):
    """
    [구현 힌트]
    1. PIL.Image.open → transform → unsqueeze(0) 으로 배치 차원 추가
    2. model(input_tensor) → torch.softmax → topk로 상위 K개 예측
    3. class_names[idx]로 클래스 이름 출력
    """
    # TODO: 이미지 로드 → 전처리 → 추론 → Top-K 결과 출력
    pass


# =============================================================================
# Step 4: 결과 시각화
# =============================================================================
"""
[시각화 옵션 - matplotlib 활용]

(1) 학습 곡선: train/val loss, accuracy를 epoch별로 plot
    - plt.subplots(1, 2) 로 loss/accuracy 나란히

(2) 예측 결과 샘플: val 배치에서 N개 이미지 + 정답/예측 표시
    - 정답이면 초록색, 오답이면 빨간색 제목
    - ⚠️ ImageNet 정규화 역변환 필요: img = img * std + mean

(3) Confusion Matrix: sklearn.metrics.confusion_matrix 활용
    - 200 클래스 전체는 복잡하므로 상위 N개만 표시 추천
"""

# TODO: 필요한 시각화 함수 구현


# =============================================================================
# Step 5: Attention Map 시각화 (Advanced - 선택)
# =============================================================================
"""
[방법 1: Forward Hook]
- WindowAttention에서 attention weight를 저장하도록 hook 등록
- register_forward_hook(hook_fn)으로 모델 수정 없이 추출 가능

[방법 2: GradCAM]
- pytorch-grad-cam 라이브러리 사용
- target_layer: 마지막 Stage의 마지막 블록의 norm 레이어
- 모델이 어디를 보고 판단하는지 히트맵으로 확인 가능
"""

# TODO: (선택) attention 시각화 구현
