# 🚀 Deep Learning Paper Implementation Sandbox (`imple_paper`)

딥러닝의 이정표가 된 주요 논문들을 직접 분석하고, 핵심 아이디어를 PyTorch로 처음부터(From Scratch) 구현하여 검증하는 개인 연구 및 학습용 저장소입니다.

단순히 코드를 복사하는 것을 넘어, 논문에서 제시하는 아키텍처 규칙을 충실히 따르고, **Tiny-ImageNet** 등의 데이터셋에서 실제로 학습과 평가를 진행하여 상세한 분석 보고서와 시각화 결과를 함께 관리합니다.

---

## 📊 구현 모델 및 상태 (Implementation Registry)

저장소에 구현된 모델들의 상세 상태 및 핵심 성과 요약입니다.

| 모델명 | 논문 정보 (Paper) | 구현 상태 (Status) | 최고 성능 (Best Accuracy) | 핵심 기술 요약 (Key Concepts) |
| :--- | :--- | :--- | :--- | :--- |
| **Swin Transformer** | [Swin Transformer (ICCV 2021)](https://arxiv.org/abs/2103.14030) | **✅ 구현 완료 (Fine-tuned)** | **Top-1 Acc: 83.16%**<br>Top-5 Acc: 95.26% | • Shifted Window 기반 Local Self-Attention<br>• Patch Merging을 통한 계층적(Hierarchical) 특징 맵 생성<br>• Relative Position Bias 적용 |
| **VGGNet** | [Very Deep CNNs (ICLR 2015)](https://arxiv.org/abs/1409.1556) | **✅ 구현 완료 (Trained)** | **Top-1 Acc: 48.67%**<br>(VGG-D, Multi-crop) | • 3×3 소형 커널을 통한 네트워크 깊이의 영향 분석<br>• Configuration A, C, D 구조 및 가중치 초기화(He)<br>• Multi-scale Training & Multi-crop(10-crop) 평가 |
| **U-Net** | [U-Net (MICCAI 2015)](https://arxiv.org/abs/1505.04597) | **✅ 모델 구현 완료** | *Architecture Only* | • Contracting Path & Expanding Path (U자형 대칭 구조)<br>• Skip Connection을 통한 고해상도 세부 특징 보존<br>• Center Crop 기반 Concat 연산 |
| **Vision Transformer (ViT)** | [An Image is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929) | **🔨 스켈레톤 및 가이드 제공** | *TODO* | • 이미지를 패치 시퀀스로 분할 후 선형 투영<br>• Class Token 및 학습 가능한 1D Positional Embedding<br>• Standard Transformer Encoder & Classification Head |

---

## 📂 프로젝트 폴더 구조 (Directory Structure)

```directory
imple_paper/
├── VGGNet/               # VGGNet 아키텍처 구현 및 실험
│   ├── model.py          # VGG Configuration A, C, D 모델 정의
│   ├── dataset.py        # Tiny-ImageNet 전처리 및 데이터 로더 (Multi-scale 지원)
│   ├── train.py          # 학습 루프 (최적 가중치 저장 및 학습률 스케줄링 적용)
│   ├── test.py           # 평가 스크립트 (Single-crop 및 10-crop 평가 지원)
│   └── result/           # 8개 실험 로그, 학습 곡선 시각화, 상세 분석 보고서
│
├── U-Net/                # U-Net 이미지 분할 모델 구현
│   ├── unet_arch.py      # UEncBlock, UDecBlock, UOutBlock 모듈화 설계 (3채널 지원)
│   └── summary-unet.txt  # 핵심 아이디어 및 학습 메커니즘 요약 메모
│
├── swin_transformer/     # Vision Transformer 계열 모델 구현 및 실험
│   ├── swin_transformer.py # W-MSA, SW-MSA, Masked Attention 직접 구현
│   ├── vit.py            # Vision Transformer 구현용 상세 가이드 및 TODO 스켈레톤
│   ├── train.py          # Swin-T Fine-tuning 스크립트 (ImageNet 가중치 활용)
│   ├── test.py           # 학습률 곡선, 클래스별 오차 분석, GradCAM 시각화 툴
│   └── result/           # 30 Epoch 학습 로그, GradCAM 시각화, 예측 오차 분석 보고서
│
├── Common/               # PyTorch 기초 학습 자료
│   └── pytorch_tutorial/ # Tensor 조작부터 선형 회귀, CNN 학습 Jupyter Notebooks (0~6)
│
├── SETUP_ENV_UV.md       # 초고속 패키지 관리자 'uv'를 이용한 개발 환경 설정 가이드
└── paper_implementation_skill.md # 논문 구현 시 뼈대 코드(Skeleton)를 만드는 설계 규칙 가이드
```

---

## ⚙️ 개발 환경 설정 및 설치 (Getting Started)

이 프로젝트는 최신적이고 매우 빠른 Python 패키지 관리 툴인 [uv](https://github.com/astral-sh/uv)를 기반으로 작성되었습니다.

### 1. 패키지 설치 및 가상환경 동기화
프로젝트 루트 디렉토리에서 다음 명령어를 실행하면, `pyproject.toml` 및 `uv.lock`에 기반한 최적의 가상환경(`python 3.14+`)이 자동으로 구축되고 라이브러리가 동기화됩니다.

```bash
# uv 설치 (설치되어 있지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 동기화 (.venv 자동 생성)
uv sync
```

### 2. 가상환경 활성화 및 실행
```bash
# 가상환경 활성화
source .venv/bin/activate

# 각 모델 디렉토리로 이동하여 스크립트 실행 가능
# 예: Swin Transformer 테스트 실행
cd swin_transformer
python test.py
```

---

## 🎯 주요 학습 목표 및 구현 원칙
1. **논문 원본 분석 우선**: 타인의 코드를 복사하는 대신, 논문의 'Methods' 섹션과 아키텍처 스펙 시트를 정밀 판독하여 직접 코딩합니다.
2. **모듈화(Modularity) 설계**: 복잡한 네트워크 구조를 가독성 높은 서브 모듈(`nn.Module`)들로 잘게 쪼개어 단계적으로 결합합니다.
3. **엄격한 텐서 크기 검증**: 각 레이어의 입력/출력 텐서 모양(Shape)을 주석에 명시하여 디버깅 비용을 최소화합니다.
4. **철저한 성능 벤치마크**: 학습 속도, 손실 수렴 양상, 일반화 능력(과적합)을 다양한 조건에서 실험하여 기록합니다.
