# VGGNet 실험 결과 분석 보고서

> 분석 대상: `VGGNet/result/` 디렉토리 내 8개 `.log` 파일  
> 데이터셋: **Tiny-ImageNet** (100,000 train / 200 classes, 입력 해상도 64×64)  
> 분석일: 2026-05-09

---

## 1. 실험 조건 요약

| 로그 파일 | 모델 | 평가 방식 | 전체 에포크 | LR 스케줄러 | BN | Best Accuracy | Best 달성 에포크 | Epoch 74 Accuracy | 최종(마지막) Accuracy | 최종 Avg Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| `train_A_single_e30.log` | VGG-A | Single-crop | 30 | 없음 | ❌ | **34.0%** | 30 (마지막) | — | 34.0% | 3.2015 |
| `train_C_single_e30.log` | VGG-C | Single-crop | 30 | 없음 | ❌ | **30.6%** | 30 (마지막) | — | 30.6% | 3.3387 |
| `train_C_single_e74.log` | VGG-C | Single-crop | 74 | 없음 | ❌ | **34.8%** | 74 (마지막) | **34.8%** | 34.8% | 4.0684 |
| `train_D_single_e30.log` | VGG-D | Single-crop | 30 | 없음 | ❌ | **36.3%** | 30 (마지막) | — | 36.3% | 3.0226 |
| `train_D_multi_e30.log` | VGG-D | Multi-crop (10개) | 30 | 없음 | ❌ | **40.9%** | 30 (마지막) | — | 40.9% | 2.5018 |
| `train_D_multi_e74.log` | VGG-D | Multi-crop (10개) | 74 | 없음 | ❌ | **48.6%** | 74 (마지막) | **48.6%** | 48.6% | 2.5495 |
| `train_ler_D_multi.log` | VGG-D | Multi-crop (10개) | 80 | ✅ (StepLR) | ❌ | **48.67%** | **Epoch 58** | 48.3% | 48.4% | 2.3647 |
| `train_ler_D_multi_BN.log` | VGG-D | Multi-crop (10개) | 80 | ✅ (StepLR) | ✅ | **39.02%** | **Epoch 75** | 38.9%¹ | 38.8% | 2.6157 |

> **Multi-crop**: 테스트 시 입력 이미지를 10개 crop으로 어그먼테이션하여 softmax 평균을 내는 방식  
> **ler (LR scheduler)**: 에포크마다 학습률을 조정 (StepLR 계열)  
> ¹ BN 실험은 5 epoch마다 test를 수행 → Epoch 70: 38.9%, Epoch 74는 test 없음 (다음 test는 Epoch 75)  

---

## 2. 구조별 성능 비교 (Single-crop, 30 Epoch)

| 모델 | Conv 레이어 수 | 특징 | Test Accuracy |
|---|---|---|---|
| VGG-A | 8 | 1×1 conv 없음, 채널 수 최소 | 34.0% |
| VGG-C | 13 | 일부 3×3 conv를 **1×1 conv로 대체** | 30.6% |
| VGG-D | 13 | 전부 **3×3 conv만 사용** (VGGNet 원 논문 구조) | 36.3% |

---

## 3. 평가 방식에 따른 성능 비교 (VGG-D, 30 Epoch)

| 평가 방식 | Test Accuracy | Avg Loss |
|---|---|---|
| Single-crop | 36.3% | 3.0226 |
| Multi-crop (10개) | 40.9% | 2.5018 |
| 차이 | **+4.6%p** | **-0.52** |

---

## 4. 에포크 수에 따른 성능 비교

| 로그 파일 | 모델 | 평가 | 에포크 | Test Accuracy |
|---|---|---|---|---|
| `train_C_single_e30.log` | VGG-C | Single | 30 | 30.6% |
| `train_C_single_e74.log` | VGG-C | Single | 74 | 34.8% |
| `train_D_multi_e30.log` | VGG-D | Multi | 30 | 40.9% |
| `train_D_multi_e74.log` | VGG-D | Multi | 74 | 48.6% |

---

## 5. LR 스케줄러 & Batch Normalization 효과 (VGG-D, Multi-crop, 80 Epoch)

| 조건 | Best Accuracy | Best 에포크 | Epoch 74 Accuracy | 최종 Accuracy | Epoch 1 초기 손실 |
|---|---|---|---|---|---|
| 고정 LR, BN 없음 | **48.6%** | 74 (마지막) | **48.6%** | 48.6% | ~5.49 |
| LR 스케줄러, BN 없음 | **48.67%** | **Epoch 58** | 48.3% | 48.4% | ~5.41 |
| LR 스케줄러, **BN 있음** | **39.02%** | **Epoch 75** | ~38.9% (Ep.70 기준) | 38.8% | **~17.18** (폭발) |

> **주목**: LR 스케줄러 실험에서 Best는 **Epoch 58**에서 이미 달성됨. 이후 58~80 epoch 동안 48.1~48.5% 사이를 진동하며 더 이상 개선되지 않음 → **조기 수렴(early convergence)** 발생

---

## 6. 결과 추론 및 분석

### 6-1. VGG-C의 성능이 VGG-A보다 낮은 이유

VGG-C는 일부 3×3 conv를 **1×1 conv로 대체**하였습니다. 1×1 conv는 공간적 특징(위치 관계)을 학습하지 못하고 채널 방향의 선형 조합만 수행합니다. Tiny-ImageNet(64×64)처럼 해상도가 낮은 데이터에서는 **공간 패턴을 포착하는 3×3 conv의 역할이 더 중요**하며, 1×1 conv 도입으로 오히려 표현력이 저하된 것으로 추론됩니다.

VGG-A(8 레이어)는 더 단순한 구조임에도 불구하고, VGG-C보다 좋은 성능을 보입니다. 이는 VGG-C의 1×1 conv 도입이 **Tiny-ImageNet 도메인에서 부정적**으로 작용했음을 시사합니다.

### 6-2. VGG-D가 가장 좋은 성능을 보이는 이유

VGG-D(VGGNet 원 논문의 핵심 구조)는 **모든 conv를 3×3으로 통일**합니다. 3×3 conv를 여러 개 쌓으면 더 큰 커널과 동일한 수용 영역(receptive field)을 가지면서도 **파라미터 수가 적고 비선형성은 더 많아** 표현력이 향상됩니다. Tiny-ImageNet처럼 다양한 클래스(200개)가 존재하는 데이터에서 이 구조가 가장 잘 작동한 것으로 보입니다.

### 6-3. Multi-crop 평가가 Single-crop보다 좋은 이유

Multi-crop은 동일한 이미지에서 **10개의 서로 다른 crop(위치/플립 조합)**을 생성한 뒤 softmax 점수를 평균냅니다. 이는 **앙상블 효과**를 통해 단일 crop의 우연적 오류를 평균화하여 더 안정적이고 정확한 예측을 가능하게 합니다. 결과적으로 같은 모델·같은 epoch 수에서 약 **+4.6%p**의 일관된 성능 향상이 관찰됩니다.

### 6-4. 에포크 수 증가가 성능 향상에 기여하는 이유

- VGG-C(30→74 epoch): 30.6% → 34.8% (**+4.2%p**)  
- VGG-D Multi-crop(30→74 epoch): 40.9% → 48.6% (**+7.7%p**)

학습이 길어질수록 모델이 **더 세밀한 특징**을 학습하게 됩니다. 특히 Tiny-ImageNet 200 클래스 분류는 복잡한 문제이므로, 충분한 학습 반복이 필요합니다. 단, VGG-C 74 epoch 로그에서 train loss는 ~0.5 수준으로 매우 낮지만 test loss는 4.07로 높아, **심각한 과적합(overfitting)이 발생**했음을 알 수 있습니다.

### 6-5. LR 스케줄러 적용이 성능을 소폭 향상시킨 이유

`train_ler_D_multi.log` (LR 스케줄러 적용)의 **Best Accuracy가 48.67%**로, 고정 LR의 74 epoch(48.6%)보다 미세하게 높습니다. LR 스케줄러는 학습 초반에는 빠른 수렴을 위해 높은 학습률을 사용하고, 후반에는 **세밀한 수렴**을 위해 학습률을 감소시킵니다. 이를 통해 **local minimum 주변에서 더 정밀하게 최적화**가 이루어지며, 최종 avg loss도 2.3647로 가장 낮습니다.

또한 LR 스케줄러 실험은 각 epoch 후 test를 수행하여 **최고 정확도 모델을 저장**(`==> New Best Accuracy`)하는 방식을 사용했으며, 이를 통해 과적합이 심화되기 전의 최적 모델을 포착할 수 있었습니다.

### 6-6. Batch Normalization이 오히려 성능을 저하시킨 이유

> ⚠️ **가장 흥미로운 결과**: LR 스케줄러 + BN 조합이 BN 없는 것보다 **약 10%p 낮은 성능**을 보임

로그 파일에서 핵심 단서는 **Epoch 1의 초기 손실값**입니다:

- BN 없음: Epoch 1 시작 손실 ≈ **5.41** (정상적인 log(200) ≈ 5.30 수준)  
- BN 있음: Epoch 1 시작 손실 ≈ **17.18** (비정상적으로 높음 → **폭발적 손실**)

이는 BN 적용 시 다음과 같은 문제가 발생했음을 시사합니다:

1. **초기화 문제**: BN 레이어의 scale(γ)/shift(β) 파라미터 초기화와 학습률(LR=0.01)이 충돌하여 초기 그래디언트가 폭발.
2. **공간 크기 문제**: 이 구현에서 classifier가 `Conv2d(512, 4096, kernel_size=(2,2))`로 되어 있어, 최종 feature map이 매우 작습니다(2×2 → 1×1). 이처럼 spatial dimension이 극히 작은 레이어에 BN을 적용하면 **batch 내 통계 추정이 불안정**해집니다.
3. **결과**: 80 epoch 내내 안정적인 수렴에 어려움을 겪어 최종 정확도가 39.02%에 그쳤으며, BN 없는 버전보다 **약 10%p 낮은 결과**를 보였습니다.

---

## 7. 종합 성능 순위 (Test Accuracy 기준)

| 순위 | 로그 파일 | Best/Final Accuracy | 핵심 조건 |
|---|---|---|---|
| 🥇 1위 | `train_ler_D_multi.log` | **48.67%** (Best) | VGG-D, Multi-crop, 80e, LR 스케줄러 |
| 🥈 2위 | `train_D_multi_e74.log` | **48.6%** | VGG-D, Multi-crop, 74e, 고정 LR |
| 🥉 3위 | `train_D_multi_e30.log` | **40.9%** | VGG-D, Multi-crop, 30e, 고정 LR |
| 4위 | `train_ler_D_multi_BN.log` | **39.02%** (Best) | VGG-D, Multi-crop, 80e, LR 스케줄러+BN |
| 5위 | `train_C_single_e74.log` | **34.8%** | VGG-C, Single-crop, 74e, 고정 LR |
| 6위 | `train_A_single_e30.log` | **34.0%** | VGG-A, Single-crop, 30e, 고정 LR |
| 7위 | `train_D_single_e30.log` | **36.3%** | VGG-D, Single-crop, 30e, 고정 LR |
| 8위 | `train_C_single_e30.log` | **30.6%** | VGG-C, Single-crop, 30e, 고정 LR |

---

## 8. 논문(~25% 에러) vs 이번 구현(~50% 에러) 성능 차이 원인

### 8-1. 비교 지표 자체가 다름

| 구분 | 논문 (VGGNet 원본) | 이번 구현 |
|---|---|---|
| 데이터셋 | **ImageNet ILSVRC** (1.2M장, **1000** 클래스) | **Tiny-ImageNet** (100K장, **200** 클래스) |
| 측정 지표 | Top-1 error **~23.7%** (= 76.3% accuracy) | Top-1 accuracy **~48.6%** (= error ~51%) |
| 입력 해상도 | **224×224** | **64×64** |

단순 수치 비교는 성립하지 않으며, 아래의 구조적 원인들이 복합적으로 작용합니다.

### 8-2. 입력 해상도 64×64 — 가장 치명적인 원인

MaxPool을 5회 통과한 후 feature map 크기:

| 입력 해상도 | MaxPool 5회 후 | 공간 위치 수 |
|---|---|---|
| **224×224** (논문) | **7×7 = 49** 위치 | 충분한 공간 표현 가능 |
| **64×64** (이번 구현) | **2×2 = 4** 위치 | 극도로 압축됨 |

Classifier 첫 레이어 `Conv2d(512, 4096, kernel_size=2×2)`가 이 4개 위치만을 입력으로 받기 때문에, 논문 대비 표현력이 극히 제한됩니다.

### 8-3. 학습 데이터 부족

| 구분 | 논문 | 이번 구현 |
|---|---|---|
| 학습 이미지 수 | **1,281,167장** | **100,000장** |
| 배수 차이 | — | 약 **12배** 적음 |

대형 모델(~138M 파라미터)은 데이터가 부족하면 과적합에 취약합니다. VGG-C 74 epoch 로그에서 train loss ~0.5, test loss 4.07로 이를 실제로 확인할 수 있습니다.

### 8-4. Multi-scale Training — 부분 구현

| 구분 | 논문 | 이번 구현 (`dataset.py`) |
|---|---|---|
| 방식 | 이미지를 **S=256~512에서 랜덤 리스케일** 후 224×224 crop | `RandomResizedCrop(64, scale=(0.5, 1.0))` |
| 효과 | 넓은 스케일 다양성 확보 | **부분적으로 유사**하나, 원본이 이미 64×64라 0.5 scale = 32×32 영역 → 매우 작은 패치 |
| 적용 실험 | 전체 | Multi 모드 실험(`isSingle=False`)에만 적용, Single 모드는 단순 `RandomCrop` |

`RandomResizedCrop`은 논문의 multi-scale과 **의도는 같지만**, 원본 해상도(64×64)의 한계로 논문만큼의 스케일 다양성을 확보하기 어렵습니다.

### 8-5. ColorJitter — 파라미터 누락으로 실질적 미적용 ❌

`dataset.py`에 `ColorJitter()`가 있지만 **파라미터를 지정하지 않아 아무 효과가 없습니다**:

```python
# 현재 구현 — 기본값이 모두 0이므로 변화 없음
transforms.ColorJitter()

# 실제 효과를 내려면
transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
```

| 파라미터 | 현재 값 | 효과 |
|---|---|---|
| `brightness` | 0 (기본값) | 밝기 변화 없음 |
| `contrast` | 0 (기본값) | 대비 변화 없음 |
| `saturation` | 0 (기본값) | 채도 변화 없음 |
| `hue` | 0 (기본값) | 색조 변화 없음 |

논문이 참조한 색상 보정(Krizhevsky et al., 2012)은 **PCA Color Augmentation**으로, 이보다 더 정교한 방식이지만 파라미터가 설정된 `ColorJitter`만으로도 일정 수준의 색상 다양성을 추가할 수 있습니다.

### 8-6. Normalize — 올바르게 구현됨 ✅

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

채널별 평균을 빼고 표준편차로 나누는 정규화로, 논문의 **RGB mean subtraction**에 해당하며 정상적으로 구현되어 있습니다.

### 8-7. 요인별 영향도 정리

| 요인 | 논문 대비 상태 | 영향도 |
|---|---|---|
| 입력 해상도 (64×64) | ❌ 불리 (7×7 → 2×2로 축소) | 🔴 매우 높음 |
| 학습 데이터 규모 | ❌ 12배 부족 | 🔴 높음 |
| Multi-scale training | ⚠️ 부분 구현 | 🟡 중간 |
| ColorJitter | ❌ 파라미터 미설정으로 미적용 | 🟡 중간 |
| Mean subtraction (Normalize) | ✅ 구현됨 | — |
| PCA Color Augmentation | ❌ 미구현 | 🟠 낮음~중간 |

---

## 9. 결론

1. **모델 구조**: VGG-D (all-3×3 conv) > VGG-A > VGG-C (1×1 conv 혼합). Tiny-ImageNet에서 1×1 conv는 오히려 역효과.
2. **평가 방식**: Multi-crop이 Single-crop 대비 일관적으로 **+4~5%p** 향상. 앙상블 효과가 명확.
3. **에포크**: 더 많은 학습은 성능 향상에 기여하나, VGG-C처럼 과적합이 심화되는 경우도 존재.
4. **LR 스케줄러**: 소폭의 성능 향상 및 더 낮은 test loss 달성. 효율적 수렴에 도움.
5. **Batch Normalization**: 현 구현에서는 **초기화 불안정** 및 작은 feature map으로 인해 역효과. LR 재조정 또는 BN 위치 재검토 필요.
6. **논문 대비 성능 차이**: 해상도(64×64)와 데이터 규모가 가장 큰 원인. `ColorJitter` 파라미터 설정으로 추가 개선 가능.
