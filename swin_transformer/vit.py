import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Step 1: 이미지를 패치로 분할하고 선형 투영(Linear Projection)을 적용하는 모듈
    """
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, img_size=224):
        super().__init__()
        # TODO: 1. 이미지를 패치로 자르고 임베딩 차원으로 매핑하는 레이어를 구현하세요.
        # (힌트: nn.Conv2d를 활용하면 매우 효율적으로 구현할 수 있습니다. stride와 kernel_size를 고려해보세요.)
        pass
        
    def forward(self, x):
        # TODO: 2. 입력 이미지 x (B, C, H, W)를 패치 시퀀스 (B, N, D)로 변환하는 과정을 구현하세요.
        # 텐서의 형태(shape)를 변경(flatten, transpose 등)해야 할 수 있습니다.
        pass

class TransformerEncoderBlock(nn.Module):
    """
    Step 3: (선택) Transformer Encoder 블록 구현
    (PyTorch의 내장 nn.TransformerEncoderLayer를 사용해도 무방합니다.)
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # TODO: 3. LayerNorm, MultiheadAttention, MLP를 초기화하세요. 
        # (ViT는 Attention과 MLP 이전에 LayerNorm을 적용하는 Pre-LN 구조입니다.)
        pass

    def forward(self, x):
        # TODO: 4. Attention -> Residual -> LayerNorm -> MLP -> Residual의 흐름을 구현하세요.
        pass

class VisionTransformer(nn.Module):
    """
    Step 2, 4, 5: 전체 아키텍처 조립
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        # 1. Patch Embedding 모듈
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        
        # TODO: 5. 학습 가능한 Class Token과 Positional Embedding 파라미터를 정의하세요. (nn.Parameter 활용)
        
        # 2. Transformer Encoder
        # TODO: 6. 앞서 만든 블록이나 nn.TransformerEncoder를 사용해 인코더를 구성하세요.
        
        # 3. Classification Head
        # TODO: 7. 분류를 위한 간단한 선형 레이어(nn.Linear)를 구현하세요.
        pass

    def forward(self, x):
        # TODO: 8. 전체 순전파(forward) 과정을 단계별로 구현하세요.
        # [흐름 가이드]
        # 1. x -> PatchEmbedding 
        # 2. Class Token을 시퀀스 맨 앞에 Concat 하기
        # 3. Positional Embedding 더하기
        # 4. Transformer Encoder 통과
        # 5. 첫 번째 위치([CLS] 토큰 위치)의 출력만 추출하기
        # 6. 추출된 토큰을 Classification Head에 통과시키기
        pass

if __name__ == "__main__":
    # 테스트용 코드 (구현하면서 주석을 해제하여 확인해보세요)
    # x = torch.randn(2, 3, 224, 224)
    # model = VisionTransformer()
    # out = model(x)
    # # print("Output shape:", out.shape) # 예상 결과: torch.Size([2, 1000])
    pass
