# U-Net

**Paper**: *U-Net: Convolutional Networks for Biomedical Image Segmentation* (Ronneberger et al., 2015)

인코더-디코더 구조와 Skip Connection을 활용하여 적은 데이터로도 정밀한 이미지 분할을 수행하는 모델을 구현합니다.

## 구현 내용

- 수축 경로(Contracting Path)와 확장 경로(Expanding Path) 아키텍처 구현
- Skip Connection을 통한 고해상도 특징 결합

## 구조

```
├── unet_arch.py       # U-Net 모델 아키텍처 정의
└── summary-unet.txt   # 논문 요약 메모
```
