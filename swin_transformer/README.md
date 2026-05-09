# Swin Transformer & ViT

Vision Transformer 계열의 모델을 구현합니다.

## Swin Transformer

**Paper**: *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., ICCV 2021)

Window 기반 Local Attention과 Shifted Window 전략으로 효율적인 계층적 비전 트랜스포머를 구현합니다.

## ViT (Vision Transformer)

**Paper**: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., ICLR 2021)

이미지를 패치 시퀀스로 변환하여 Transformer 인코더에 입력하는 구조를 구현합니다.

## 구조

```
├── swin_transformer.py  # Swin Transformer 구현
├── vit.py               # Vision Transformer 구현
├── train.py             # 학습
└── test.py              # 평가
```
