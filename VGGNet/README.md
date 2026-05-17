# 📐 VGGNet: Very Deep Convolutional Networks for Image Recognition

**Paper**: *Very Deep Convolutional Networks for Large-Scale Image Recognition* (Simonyan & Zisserman, ICLR 2015)  
**Dataset**: **Tiny-ImageNet** (200 Classes, 100,000 Train / 10,000 Val, 64×64 Resolution)  
**Implementation**: PyTorch (From Scratch)  
**Status**: **✅ 구현 및 비교 실험 완료**

---

## 💡 VGGNet 핵심 요약

VGGNet은 **"네트워크의 깊이(Depth)가 모델 표현력에 미치는 영향"**을 체계적으로 규명한 논문입니다. 기존 AlexNet 등에서 사용하던 대형 필터(11×11, 7×7) 대신, **3×3 소형 컨볼루션 필터**만을 연속적으로 깊게 쌓아 다음과 같은 이점을 증명했습니다.

1. **비선형성 증가**: 3×3 conv를 3개 쌓으면 7×7 conv 1개와 동일한 수용 영역(Receptive Field)을 가지면서도, 활성화 함수(ReLU)를 3번 거쳐 더 복잡하고 풍부한 비선형 관계를 학습합니다.
2. **파라미터 수 감소**: 3×3 conv 3개의 채널 수가 $C$일 때 파라미터는 $3 \times (3^2 \times C^2) = 27C^2$로, 7×7 conv 1개의 파라미터인 $1 \times (7^2 \times C^2) = 49C^2$ 대비 약 **45% 절감**됩니다.

---

## 📂 디렉토리 구조 및 소스 역할

```directory
VGGNet/
├── model.py           # VGG Configuration A(11레이어), C(13레이어), D(16레이어) 모델 정의
├── dataset.py         # Tiny-ImageNet 전처리 (Single-crop 및 Multi-scale RandomResizedCrop 지원)
├── train.py           # 기본 고정 LR 학습 엔진 (최고 성능 가중치 저장 기능 탑재)
├── test.py            # 기본 평가 엔진 (Single-crop 및 10-crop Multi-crop 평가 모드 탑재)
├── ler_model.py       # 학습률 스케줄러 & Batch Normalization 대응을 위한 모델 정의
├── ler_train.py       # StepLR 학습률 스케줄러가 결합된 학습 루프
├── ler_test.py        # 학습률 스케줄링 모델 전용 평가 스크립트
├── ler_main.py        # LR Scheduler + BN 실험 통합 실행 메인 진입점
└── result/            # 8개 실험 로그 파일(.log), 학습 손실 그래프(.png), 상세 분석 보고서
```

---

## 📊 종합 실험 결과 요약 (Tiny-ImageNet 벤치마크)

다양한 모델 설정, 평가 기법, 하이퍼파라미터 튜닝에 따른 8가지 실험 결과입니다.

| 순위 | 실험 로그 파일명 | 아키텍처 | 에포크 | LR 스케줄러 | Batch Norm | 평가 방식 | 최고 정확도 (Best Acc) | 특징 및 수렴 분석 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1위** | `train_ler_D_multi.log` | **VGG-D** | 80 | **✅ StepLR** | ❌ | **10-Crop** | **48.67%** | Epoch 58 조기 수렴, 최저 손실(2.36) 달성 |
| 🥈 **2위** | `train_D_multi_e74.log` | **VGG-D** | 74 | ❌ 고정 LR | ❌ | **10-Crop** | **48.60%** | 후반부 학습 정체 극복, 미세 성능 향상 |
| 🥉 **3위** | `train_D_multi_e30.log` | **VGG-D** | 30 | ❌ 고정 LR | ❌ | **10-Crop** | **40.90%** | Single-crop 대비 앙상블 효과로 대폭 상승 |
| **4위** | `train_ler_D_multi_BN.log` | **VGG-D** | 80 | **✅ StepLR** | **✅ 적용** | **10-Crop** | **39.02%** | 초기 그래디언트 불안정으로 성능 저하 |
| **5위** | `train_D_single_e30.log` | **VGG-D** | 30 | ❌ 고정 LR | ❌ | Single-Crop | **36.30%** | 30 epoch 기준 단일 크롭 최고치 |
| **6위** | `train_C_single_e74.log` | **VGG-C** | 74 | ❌ 고정 LR | ❌ | Single-Crop | **34.80%** | Train Loss 0.5 미만이나 Test Loss 폭발(과적합) |
| **7위** | `train_A_single_e30.log` | **VGG-A** | 30 | ❌ 고정 LR | ❌ | Single-Crop | **34.00%** | 단순한 VGG-A 구조가 VGG-C보다 우세 |
| **8위** | `train_C_single_e30.log` | **VGG-C** | 30 | ❌ 고정 LR | ❌ | Single-Crop | **30.60%** | 1×1 conv 사용으로 공간 패턴 보존 실패 |

---

## 🔍 핵심 연구 분석 및 인사이트

### 1. VGG-D(16 레이어) vs VGG-C(13 레이어 with 1x1 Conv)
* **발견**: VGG-D는 30 Epoch에서 **36.3%**를 기록한 반면, 일부 3×3 conv를 1×1 conv로 대체한 VGG-C는 **30.6%**로 성능이 하락했습니다. 심지어 더 얕고 단순한 VGG-A(34.0%)보다도 낮습니다.
* **이유**: 1×1 conv는 채널 방향의 조합만 제공할 뿐 공간 정보(Spatial context)를 보존하지 못합니다. Tiny-ImageNet과 같이 해상도가 매우 낮은(64×64) 환경에서는 극도로 축소된 피처 맵의 공간 기하 패턴을 유지하는 **3×3 conv의 역할이 결정적**이었음을 입증합니다.

### 2. Single-crop vs Multi-crop (10-Crop) 평가 효과
* **발견**: 동일한 VGG-D 30 Epoch 학습 가중치 조건에서 평가 방식을 Multi-crop으로 전환한 것만으로 성능이 **36.3% ➔ 40.9% (+4.6%p)** 향상되었습니다.
* **이유**: 이미지를 좌우 반전 및 4개 모서리+중앙 등 총 10가지 영역으로 자른 뒤 추론하여 Softmax 스코어를 평균내는 앙상블 효과를 거둡니다. 단일 크롭의 위치적 왜곡이나 무작위 노이즈를 상쇄하여 안정적인 정밀 분류를 수행하게 됩니다.

### 3. 학습률 스케줄러(StepLR)의 도입 성과
* **발견**: 고정 학습률(LR)의 경우 74 Epoch까지 서서히 증가해 48.60%에 도달했으나, StepLR 학습률 스케줄링을 연계하자 **단 58 Epoch 만에 최고 정확도인 48.67%**를 기록하며 수렴 안정성이 비약적으로 증가했습니다.

### 4. ⚠️ Batch Normalization의 역효과 및 분석
* **이유**: LR 스케줄러와 Batch Normalization을 동시 연계한 실험에서 정확도가 **39.02%**로 오히려 급격히 후퇴했습니다. 로그 분석 결과 두 가지 핵심적 병목이 포착되었습니다:
  1. **초기 가중치 충돌**: Epoch 1의 첫 배치 손실값이 BN 미적용 시 약 `5.41`로 정상 범위(log(200) ≈ 5.3)인 반면, BN 적용 시 **`17.18`로 폭발**하였습니다. 가중치 초기화 분산과 배치 정규화 스케일 파라미터 간 불일치가 발생했기 때문입니다.
  2. **극단적 공간 해상도**: 본 아키텍처는 64×64 입력 기준 5회의 MaxPool을 수행하여 최종 해상도가 `2×2 = 4` 공간 위치만 남습니다. 이 극소 해상도 단계에 배치 정규화를 통과시키면 배치 내 채널 평균/분산 통계치 추정이 극도로 왜곡(Vanishing/Exploding Statistics)되어 오히려 학습 안정성을 붕괴시키는 것으로 분석됩니다.

---

## 🛠️ 실행 및 재현 가이드 (Commands)

### 1. 데이터 준비 및 가상환경 활성화
의존성 라이브러리가 완비된 가상환경을 동기화하고 실행합니다.
```bash
source ../.venv/bin/activate
```

### 2. VGGNet 고정 LR 학습 실행 (Configuration A, C, D 선택 가능)
`train.py` 내의 `model = VGGNet_D()` 파트를 변경하여 실험할 수 있습니다.
```bash
python train.py
```

### 3. Multi-crop 평가 수행
학습된 최적 가중치 `result/*.pth` 파일을 활용하여 10-crop 성능을 측정합니다.
```bash
python test.py
```

### 4. 학습률 스케줄러 & BN 연계 실험 실행
```bash
# 메인 실험 시작 (ler_model.py, ler_train.py 연계)
python ler_main.py

# 스케줄러 적용 모델 평가
python ler_test.py
```

---

## 📌 원 논문(ImageNet) 대비 성능 차이 원인 분석

VGGNet 원 논문의 ImageNet Top-1 정확도(~76.3%)와 본 실험의 Tiny-ImageNet 정확도(~48.67%) 간 격차는 다음 세 요인에 기인합니다.

* **입력 해상도 제약**: 224×224 입력 시 5회 MaxPool 후에도 7×7(=49) 공간 표현이 유지되나, 64×64 입력 시에는 2×2(=4) 공간 표현으로 줄어들어 정보 압축 손실이 매우 심각합니다.
* **데이터 규모 부족**: ImageNet(128만 장) 대비 Tiny-ImageNet(10만 장)은 약 12배 부족하여 파라미터가 1억 개를 상회하는 VGGNet이 특징점을 일반화하지 못하고 과적합(Overfitting)되기 쉽습니다.
* **ColorJitter 설정 미정의**: `dataset.py` 내부 트랜스폼 정의 시 `transforms.ColorJitter()`와 같이 파라미터를 누락하여 기본값(모든 변경 폭 0)이 주어져 색상 변조 데이터 보강(Augmentation) 효과가 실질적으로 누락되었습니다. 향후 `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)` 형태로 갱신하여 추가 성능 향상을 꾀할 수 있습니다.
