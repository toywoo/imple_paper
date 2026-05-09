# VGGNet

**Paper**: *Very Deep Convolutional Networks for Large-Scale Image Recognition* (Simonyan & Zisserman, 2014)

3×3 컨볼루션을 깊게 쌓아 네트워크 깊이가 성능에 미치는 영향을 탐구한 논문을 구현합니다.

## 구현 내용

- VGG Configuration A, C, D 아키텍처 구현
- Tiny-ImageNet 데이터셋 기반 학습 및 평가
- Single-scale / Multi-scale 평가 비교 실험
- 학습률 스케줄링 적용 실험

## 구조

```
├── model.py       # VGGNet 모델 정의
├── dataset.py     # 데이터 로딩 및 전처리
├── train.py       # 학습 루프
├── test.py        # 평가
└── result/        # 학습 로그 및 시각화
```
