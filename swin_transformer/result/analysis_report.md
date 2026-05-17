# Swin Transformer 학습 결과 분석 보고서

본 보고서는 `swin_transformer/train.log` 파일에 기록된 30 Epoch 동안의 Swin Transformer 모델 학습 결과를 바탕으로 작성되었습니다.
추가로 이전 실험인 **VGGNet**(`VGGNet/result/analysis_report.md`) 실험 결과와의 비교 분석을 포함합니다.

## 1. 학습 개요
- **사용 기기 (Device):** CUDA (GPU)
- **총 학습 에포크 (Epochs):** 30
- **배치 수 (Batches per Epoch):** 1563
- **에포크 당 소요 시간:** 평균 약 1165초 (약 19.4분)
- **총 학습 시간:** 약 9.7시간

## 2. 주요 성능 지표 요약

| 지표 (Metrics) | 초기 (Epoch 1) | 최종 (Epoch 30) | 최고 성능 (Best) |
| --- | --- | --- | --- |
| **Train Loss** | 2.2945 | 0.9386 | **0.9386** (Epoch 30) |
| **Train Accuracy** | 61.77% | 98.84% | **98.84%** (Epoch 30) |
| **Validation Loss** | 1.5958 | 1.4556 | **1.4542** (Epoch 28) |
| **Val Top-1 Accuracy** | 77.79% | 82.96% | **83.16%** (Epoch 28) |
| **Val Top-5 Accuracy** | 93.58% | 95.25% | **95.26%** (Epoch 28) |

## 3. 학습 트렌드 분석

### 3.1. Training (학습 데이터셋)
- **Loss 감소:** 초기 2.29에서 최종 0.93 수준으로 매우 안정적이고 꾸준하게 감소했습니다.
- **Accuracy 증가:** 첫 Epoch에서 61.77%로 시작하여 최종 30 Epoch에서는 98.84%에 도달했습니다. 모델이 학습 데이터의 특징을 매우 훌륭하게 학습하고 있음을 보여줍니다.

### 3.2. Validation (검증 데이터셋)
- **최고 성능 도달:** Validation Top-1 Accuracy는 점진적으로 상승하여 **Epoch 28에서 최고치인 83.16%**를 기록했습니다. 
- **과적합 (Overfitting) 경향성:**
  - Training Accuracy는 거의 99%에 육박하지만, Validation Accuracy는 약 83% 대에서 정체되는 모습을 보입니다. (Epoch 24 이후 약 82.8% ~ 83.1% 사이에서 등락)
  - Validation Loss는 꾸준히 1.4 ~ 1.5 구간에 머물러 있으며, Training Loss처럼 큰 폭으로 떨어지지 않고 있습니다. 
  - 전형적인 딥러닝 모델의 일반화 한계점(Generalization Gap)을 보여주며 다소간의 과적합(Overfitting)이 발생하고 있는 것으로 추정됩니다.

---

## 4. 🚀 VGGNet과의 성능 비교 분석

이전 Tiny-ImageNet 데이터셋에서 진행된 VGGNet 실험 결과와 이번 Swin Transformer 결과를 비교한 내용은 다음과 같습니다.

### 4.1. 압도적인 성능 차이
| 모델 | Best Val Top-1 Acc | 학습 에포크 수 | 특이사항 |
| --- | --- | --- | --- |
| **VGGNet (최고 성능: VGG-D)** | **48.67%** | 80 Epoch | Multi-crop, LR Scheduler 적용 |
| **Swin Transformer** | **83.16%** | 30 Epoch | 단 **1 Epoch** 만에 77.79% 달성 |

- **성능 향상:** Swin Transformer는 VGGNet 대비 **+34.49%p** 라는 압도적인 정확도 향상을 보여주었습니다.
- **수렴 속도:** VGGNet은 80 Epoch를 학습하여 48.67%에 도달한 반면, Swin Transformer는 단 **1 Epoch 만에 77.79%**를 기록했습니다. 이는 사전 학습(Pre-trained) 가중치를 활용한 전이 학습(Transfer Learning)의 강력함과, Vision Transformer 아키텍처 자체의 우수성을 명확히 보여줍니다.

### 4.2. 아키텍처 특성에 따른 해상도 극복
- **VGGNet의 한계:** VGGNet은 64x64 입력 해상도에서 5번의 MaxPool을 거치며 특징 맵이 2x2 사이즈로 극도로 축소되어, 공간적 정보를 소실하는 치명적인 한계가 있었습니다.
- **Swin Transformer의 장점:** Swin Transformer는 Shifted Window 기반의 Self-Attention을 통해 국소적(Local) 정보뿐만 아니라 전역적(Global) 컨텍스트를 효과적으로 포착합니다. 패치 병합(Patch Merging) 과정을 거치더라도 Self-Attention 메커니즘 덕분에 VGGNet과 같은 심각한 공간 정보 손실을 겪지 않고, 64x64 해상도에서도 훌륭한 표현력을 유지합니다.

### 4.3. 일반화 성능 (과적합 방어력)
- **VGGNet:** `Train Loss`가 0.5 수준으로 떨어졌음에도 `Test Loss`가 4.07에 달하는 등 심각한 과적합 현상을 보였습니다. 대량의 파라미터(약 138M)를 가졌음에도 작은 데이터셋(10만 장)에서 특징을 제대로 일반화하지 못했습니다.
- **Swin Transformer:** 역시 `Train Acc` 98.84%와 `Val Acc` 83.16%로 과적합 경향은 존재하지만, Validation Loss가 1.45 수준에서 방어되고 있습니다. VGGNet 대비 월등히 안정적이고 뛰어난 일반화 능력을 보여줍니다.

---

## 5. 시사점 및 향후 개선 방향

1. **Transformer 기반 모델의 우수성 입증:**
   - 기존 CNN(VGGNet) 아키텍처 대비 현대적인 Transformer 구조(Swin)가 동일 데이터셋에서 얼마나 큰 격차를 낼 수 있는지 증명되었습니다.
   
2. **과적합(Overfitting) 완화 전략 (추가 개선 시):**
   - 현재 모델도 83% 대의 훌륭한 성능을 보이지만 99%에 가까운 Train Acc를 고려할 때, **Data Augmentation(Mixup, CutMix, RandAugment)**이나 **Regularization(DropPath, Weight Decay 비율 조절)** 기법을 추가하면 VGGNet 실험처럼 큰 폭의 앙상블/일반화 성능 향상(+4~5%p)을 얻을 잠재력이 큽니다.
   - 이미 28 Epoch에서 최고 성능을 달성하므로, 무리한 Epoch 연장보다는 조기 종료(Early Stopping)와 하이퍼파라미터 튜닝에 집중하는 것이 효과적입니다.

**결론:** Swin Transformer 실험은 VGGNet 대비 **+34.49%p 정확도 상승**이라는 경이로운 성과를 거두었으며, 아키텍처의 세대 교체와 전이 학습의 위력을 보여주는 매우 성공적인 실험이었습니다.
