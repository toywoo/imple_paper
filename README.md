# 📄 imple_paper

딥러닝 주요 논문들을 직접 읽고, 핵심 아이디어를 코드로 구현해보는 저장소입니다.

논문의 아키텍처와 학습 방식을 최대한 따라가되, 이해와 실험을 목적으로 합니다.

## 구현 목록

| 모델 | 논문 | 상태 |
|------|------|------|
| VGGNet | Very Deep Convolutional Networks for Large-Scale Image Recognition (2014) | ✅ 구현 완료 |
| U-Net | U-Net: Convolutional Networks for Biomedical Image Segmentation (2015) | ✅ 모델만 구현 완료 |
| Swin Transformer | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021) | 🔨 구현 중 |
| ViT | An Image is Worth 16x16 Words (2020) | 🔨 구현 중 |

## 프로젝트 구조

```
├── VGGNet/          # VGGNet 구현 및 실험 결과
├── U-Net/           # U-Net 아키텍처 구현
├── swin_transformer/ # Swin Transformer & ViT 구현
├── Common/          # PyTorch 기초 학습 자료
└── SETUP_ENV_UV.md  # 환경 설정 가이드
```

## 환경

- Python 3.14+
- PyTorch
- 패키지 관리: [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```
