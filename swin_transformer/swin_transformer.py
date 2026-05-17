"""
=============================================================================
Swin Transformer 구현 가이드
Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
       (Liu et al., ICCV 2021)
=============================================================================

[핵심 아이디어]
- ViT는 모든 패치 간에 Global Self-Attention을 수행 → 계산량이 이미지 크기에 대해 O(N²)
- Swin Transformer는 **Window 기반 Local Attention** + **Shifted Window** 전략으로
  O(N)에 가까운 선형 복잡도를 달성하면서도 cross-window 정보 교환을 가능하게 함
- **계층적(Hierarchical) 구조**: Stage마다 Patch Merging으로 해상도를 줄이고 채널을 늘림
  → CNN의 FPN처럼 다양한 스케일의 feature map 생성 가능

[전체 아키텍처 흐름]
  Input Image (B, 3, H, W)
       ↓
  ① Patch Partition + Linear Embedding  →  (B, H/4 * W/4, C)
       ↓
  ② Stage 1: Swin Transformer Block × 2  →  (B, H/4 * W/4, C)
       ↓
  ③ Patch Merging                        →  (B, H/8 * W/8, 2C)
       ↓
  ④ Stage 2: Swin Transformer Block × 2  →  (B, H/8 * W/8, 2C)
       ↓
  ⑤ Patch Merging                        →  (B, H/16 * W/16, 4C)
       ↓
  ⑥ Stage 3: Swin Transformer Block × 6  →  (B, H/16 * W/16, 4C)
       ↓
  ⑦ Patch Merging                        →  (B, H/32 * W/32, 8C)
       ↓
  ⑧ Stage 4: Swin Transformer Block × 2  →  (B, H/32 * W/32, 8C)
       ↓
  ⑨ Global Average Pooling + Classifier  →  (B, num_classes)

[Swin-Tiny(Swin-T) 기본 하이퍼파라미터]
  - C = 96 (초기 임베딩 차원)
  - 각 Stage의 블록 수: [2, 2, 6, 2]
  - 각 Stage의 Attention Head 수: [3, 6, 12, 24]
  - Window size: 7×7
  - Patch size: 4×4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Step 1: Patch Embedding (Patch Partition + Linear Embedding)
# =============================================================================
class PatchEmbedding(nn.Module):
    """
    이미지를 4×4 패치로 분할하고 선형 투영하는 모듈.

    [ViT와의 차이점]
    - ViT: 16×16 패치, 패치 수가 적음 (196개)
    - Swin: 4×4 패치, 패치 수가 많음 (3136개) → 이후 Patch Merging으로 점진적 축소

    [구현 힌트]
    - nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
      를 사용하면 패치 분할과 선형 투영을 한번에 처리할 수 있음
    - 출력을 (B, N, C) 형태로 reshape 해야 함 (N = H/4 * W/4)
    - LayerNorm 적용하는 것을 권장 (논문에서 사용)
    """
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        # TODO: Conv2d로 패치 분할 + 선형 투영 구현
        # TODO: LayerNorm(embed_dim) 추가
        self.conv = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2, 3)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - 입력 이미지
        Returns:
            (B, H/4 * W/4, embed_dim) - 패치 임베딩, H와 W도 함께 반환하면 편리
        """
        # TODO: Conv2d → flatten → transpose → LayerNorm
        x = self.conv(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        return x


# =============================================================================
# Step 2: Patch Merging
# =============================================================================
class PatchMerging(nn.Module):
    """
    Stage 사이에 해상도를 절반으로 줄이고 채널을 2배로 늘리는 다운샘플링 모듈.

    [동작 원리]
    - 입력: (B, H*W, C)
    - 2×2 이웃 패치를 하나로 합침:
      x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2] 를 concat → (B, H/2*W/2, 4C)
    - nn.Linear(4C, 2C) 로 채널 수 조정 → 최종 (B, H/2*W/2, 2C)

    [구현 힌트]
    - 먼저 (B, H*W, C) → (B, H, W, C) 로 reshape
    - 인덱싱으로 4개 그룹 추출 후 torch.cat(dim=-1)
    - LayerNorm → Linear 적용
    """
    def __init__(self, dim):
        super().__init__()
        # TODO: LayerNorm(4 * dim), Linear(4 * dim, 2 * dim, bias=False) 정의
        self.ln = nn.LayerNorm(4 * dim)
        self.fc = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: 현재 feature map의 높이, 너비
        Returns:
            (B, H/2 * W/2, 2C), 새로운 H, W
        """
        # TODO: reshape → 2×2 그룹핑 → concat → norm → linear
     
        x = x.reshape((x.shape[0], H, W, x.shape[2]))
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = self.ln(x)
        x = self.fc(x)

        x = x.view(x.shape[0], -1, x.shape[-1])

        return x, H // 2, W // 2


# =============================================================================
# Step 3: Window Partition / Reverse (유틸리티 함수)
# =============================================================================
def window_partition(x, window_size):
    """
    Feature map을 window_size × window_size 크기의 윈도우들로 분할.

    [구현 힌트]
    - 입력: (B, H, W, C)
    - reshape: (B, H//ws, ws, W//ws, ws, C)
    - permute: (B, H//ws, W//ws, ws, ws, C)
    - reshape: (B * num_windows, ws, ws, C)
      여기서 num_windows = (H//ws) * (W//ws)

    Args:
        x: (B, H, W, C)
        window_size: int (예: 7)
    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    # TODO: 위 힌트대로 view, permute, contiguous, view 순서로 구현
    B, H, W, C = x.shape
    ws = window_size
    x = x.view(B, H//ws, ws, W//ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, ws, ws, C)
    return x



def window_reverse(windows, window_size, H, W):
    """
    window_partition의 역연산. 분할된 윈도우들을 다시 원래 feature map으로 합침.

    [구현 힌트]
    - window_partition의 reshape/permute를 역순으로 수행

    Args:
        windows: (B * num_windows, window_size, window_size, C)
        window_size: int
        H, W: 원래 feature map 크기
    Returns:
        x: (B, H, W, C)
    """
    # TODO: window_partition의 역과정 구현
    ws = window_size
    _B, _H, _W, C = windows.shape
    B = _B // (H//ws * W//ws)
    x = windows.view(B, H//ws, W//ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)

    return x



# =============================================================================
# Step 4: Window-based Multi-Head Self Attention (W-MSA)
# =============================================================================
class WindowAttention(nn.Module):
    """
    윈도우 내에서만 수행하는 Multi-Head Self Attention.

    [핵심 개념 - Relative Position Bias]
    - ViT처럼 absolute position embedding을 쓰지 않음
    - 대신 각 head마다 relative position bias B를 attention 점수에 더함:
      Attention(Q,K,V) = softmax(QK^T / √d + B) V
    - B는 학습 가능한 파라미터로, relative_position_bias_table에서 인덱싱

    [구현 힌트]
    1. __init__에서:
       - qkv = nn.Linear(dim, dim * 3) 으로 Q, K, V를 한번에 생성
       - relative_position_bias_table: nn.Parameter, shape = ((2*ws-1)*(2*ws-1), num_heads)
       - relative_position_index: register_buffer로 등록 (학습 안 되는 인덱스 테이블)
         → 이 인덱스 계산이 조금 복잡한데, 아래 참고:
           coords_h = torch.arange(ws)
           coords_w = torch.arange(ws)
           coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, ws, ws)
           coords_flatten = coords.view(2, -1)                          # (2, ws*ws)
           relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws*ws, ws*ws)
           relative_coords를 적절히 변환하여 1D 인덱스로 만듦
       - proj = nn.Linear(dim, dim) 으로 출력 투영

    2. forward에서:
       - qkv 계산 후 (B*nW, N, 3, nH, C//nH) 로 reshape, permute
       - attention score = q @ k.transpose(-2,-1) * scale
       - relative_position_bias를 더함
       - (선택) attention mask가 있으면 적용 (Shifted Window에서 사용)
       - softmax → dropout → @ v → reshape → proj
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        # TODO: 위 힌트를 참고하여 qkv, proj, relative_position_bias_table 등 정의
        # TODO: relative_position_index를 계산하고 register_buffer로 등록
        ws = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(((2*ws-1)*(2*ws-1), num_heads)))
        
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, ws, ws)
        coords_flatten = coords.view(2, -1)                          # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws*ws, ws*ws)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (ws*ws, ws*ws, 2)
        relative_coords[:, :, 0] += ws - 1 
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout()
        self.proj = nn.Linear(dim, dim) 


    def forward(self, x, mask=None):
        """
        Args:
            x: (B*num_windows, window_size*window_size, C)
            mask: (num_windows, ws*ws, ws*ws) or None
                  - None이면 일반 W-MSA
                  - 값이 있으면 SW-MSA용 attention mask
        Returns:
            (B*num_windows, window_size*window_size, C)
        """
        # TODO: qkv → attention score + relative position bias (+ mask) → softmax → value → proj
        B_nW, N, C = x.shape
        nH = self.num_heads

        qkv = self.qkv(x) # (B*num_windows, window_size*window_size, 3*C) -> (B*nW, N, 3, nH, C//nH)
        qkv = qkv.reshape(B_nW, N, 3, nH, C // nH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # #3 (B*nW, nH, N, C//nH)

        scale = (C // nH) ** 0.5
        attention_score = torch.matmul(q, k.transpose(-2,-1)) / scale # (B * num_windows, nH, N, N)
        relative_bias = self.relative_position_bias_table[self.relative_position_index].permute(2, 0, 1) 
        attention_score += relative_bias

        if mask is not None:
            nW = mask.shape[0]
            attention_score = attention_score.view(-1, nW, nH, N, N) # (B, nW, nH, N, N)
            attention_score += mask.unsqueeze(1).unsqueeze(0) # (num_windows, ws*ws, ws*ws) -> (nW, 1, N, N) -> (1, nW, 1, N, N)
            attention_score = attention_score.view(-1, nH, N, N) # (B * num_windows, nH, N, N)

        attention_score = self.softmax(attention_score)
        attention_score = self.dropout(attention_score)
        out = torch.matmul(attention_score, v)
        out = out.permute(0, 2, 1, 3).reshape(B_nW, N, C)
        out = self.proj(out)

        return out


# =============================================================================
# Step 5: Swin Transformer Block
# =============================================================================
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer의 핵심 블록. W-MSA 또는 SW-MSA를 수행.

    [핵심 개념 - Shifted Window]
    - 연속된 두 블록이 하나의 쌍을 이룸:
      Block 1: 일반 Window Partition (W-MSA)
      Block 2: window_size//2 만큼 이동한 Shifted Window (SW-MSA)
    - Shifted Window는 torch.roll()로 구현:
      → 이동 후 window partition → attention(with mask) → window reverse → 역이동

    [Attention Mask 생성 (SW-MSA용)]
    - shift 후 window를 나누면, 인접하지 않은 영역이 같은 윈도우에 포함될 수 있음
    - 이를 방지하기 위해 attention mask를 만들어 해당 위치의 attention을 -100 등으로 차단
    - mask 생성 방법:
      1. (1, H, W, 1) 크기의 텐서에 각 영역에 0~8 등의 ID 부여
      2. window_partition 후 같은 윈도우 내 다른 ID끼리의 attention을 masking

    [구현 힌트]
    - __init__: LayerNorm ×2, WindowAttention, MLP(2-layer with GELU), shift_size 설정
    - forward:
      1. shortcut = x
      2. LayerNorm
      3. (B, H*W, C) → (B, H, W, C) reshape
      4. shift_size > 0이면 torch.roll로 이동
      5. window_partition → WindowAttention(mask) → window_reverse
      6. shift_size > 0이면 torch.roll로 역이동
      7. (B, H, W, C) → (B, H*W, C) reshape
      8. x = shortcut + x (Residual)
      9. x = x + MLP(LayerNorm(x)) (Residual)
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # TODO: LayerNorm, WindowAttention, MLP 정의
        # TODO: shift_size > 0일 때 attention mask를 미리 계산하고 register_buffer로 등록하는 것을 고려
        #       (또는 forward에서 동적으로 생성해도 됨)
        self.pre_ln = nn.LayerNorm(dim)
        self.post_ln = nn.LayerNorm(dim)

        self.window_attn = WindowAttention(dim, window_size, num_heads)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.shift_size = shift_size
        self.window_size = window_size

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: feature map 크기
        Returns:
            (B, H*W, C)
        """
        # TODO: 위 구현 힌트의 1~9 단계를 따라 구현
        B, _HW, C = x.shape
        shortcut = x
        x = self.pre_ln(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # 1. 원본 피처맵과 똑같은 크기의 가짜 맵을 만듭니다.
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            
            # 2. 이동 후 쪼개졌을 때 서로 다른 구역이 될 곳에 고유 번호(0~8)를 매깁니다.
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # 3. 이 가짜 맵도 똑같이 윈도우 크기로 쪼갭니다.
            mask_windows = window_partition(img_mask, self.window_size)  # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (nW, ws*ws)
            
            # 4. 윈도우 안의 패치들끼리 번호가 같은지 다른지 확인합니다.
            # shape: (nW, ws*ws, ws*ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            
            # 번호가 다르면(0이 아니면) -100을 더해 차단하고, 같으면 0을 더해 유지합니다.
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        
        x_win = window_partition(x, self.window_size) #  (B * num_windows, window_size, window_size, C)
        x_win = x_win.view(-1, self.window_size * self.window_size, C)
        x_win = self.window_attn(x_win, mask=attn_mask) # (B*num_windows, window_size*window_size, C) -> (B*num_windows, window_size*window_size, C)
        x_win = x_win.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_win, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.post_ln(x))

        return x

# =============================================================================
# Step 6: Swin Transformer Stage
# =============================================================================
class BasicLayer(nn.Module):
    """
    하나의 Stage를 구성. 여러 개의 SwinTransformerBlock + (선택적) PatchMerging.

    [구현 힌트]
    - depth개의 SwinTransformerBlock을 nn.ModuleList로 생성
    - 짝수 인덱스(0, 2, 4, ...): shift_size=0  (W-MSA)
    - 홀수 인덱스(1, 3, 5, ...): shift_size=window_size//2  (SW-MSA)
    - 마지막 Stage가 아니면 PatchMerging을 붙임 (downsample)
    """
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, downsample=True):
        super().__init__()
        # TODO: SwinTransformerBlock × depth개 생성 (shift_size 교대)
        # TODO: downsample=True이면 PatchMerging 추가
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else (window_size // 2)

            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, 
                    num_heads=num_heads, 
                    window_size=window_size, 
                    shift_size=shift_size, 
                    mlp_ratio=mlp_ratio)
            )

        if downsample is True:
            self.patch_merger = PatchMerging(dim)

        self.downsample = downsample

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: 현재 feature map 크기
        Returns:
            x: (B, H'*W', C') - downsample 적용 시 크기 변화
            H', W': 새 feature map 크기
        """
        # TODO: 각 블록 순회 → (downsample 있으면) PatchMerging 적용
        for block in self.blocks:
            x = block(x, H, W)
        
        if self.downsample:
            x, H, W = self.patch_merger(x, H, W)

        return x, H, W


# =============================================================================
# Step 7: 전체 Swin Transformer 모델
# =============================================================================
class SwinTransformer(nn.Module):
    """
    전체 Swin Transformer 아키텍처를 조립하는 메인 클래스.

    [Swin-T 기본 설정]
    - embed_dim = 96
    - depths = [2, 2, 6, 2]
    - num_heads = [3, 6, 12, 24]
    - window_size = 7
    - num_classes = 1000 (ImageNet) → fine-tune 시 200 (Tiny-ImageNet)
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0):
        super().__init__()
        # TODO: 1. PatchEmbedding 생성
        # TODO: 2. 4개의 BasicLayer(Stage) 생성
        #          - Stage i의 dim = embed_dim * 2^i
        #          - 마지막 Stage에는 downsample=False
        # TODO: 3. LayerNorm(최종 채널 수)
        # TODO: 4. AdaptiveAvgPool1d(1) 또는 단순 mean으로 Global Average Pooling
        # TODO: 5. nn.Linear(최종 채널 수, num_classes) 분류 헤드
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.ln = nn.LayerNorm(final_dim)

        self.basic_layers = nn.ModuleList()
        for i in range(4):
            downsample = True
            if i == 3:
                downsample = False
            self.basic_layers.append(BasicLayer(embed_dim * 2 ** i, depths[i], num_heads[i], window_size, mlp_ratio, downsample))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.final_fc = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, num_classes)
        """
        # TODO: PatchEmbedding → Stage 1~4 → LayerNorm → GAP → Classifier
        B, C, H, W = x.shape
        _H = H // self.patch_size
        _W = W // self.patch_size

        x = self.patch_embedding(x)
        
        for basic_layer in self.basic_layers:
            x, _H, _W = basic_layer(x, _H, _W)
        
        x = self.ln(x)

        x = x.transpose(1, 2)
        x = self.avg_pool(x)
        x = x.flatten(1)

        x = self.final_fc(x)

        return x


# =============================================================================
# 테스트 코드 (구현 완료 후 주석 해제하여 확인)
# =============================================================================
if __name__ == "__main__":
    # --- 기본 동작 테스트 ---
    x = torch.randn(2, 3, 224, 224)
    model = SwinTransformer(num_classes=200)  # Tiny-ImageNet용
    out = model(x)
    print("Output shape:", out.shape)  # 예상: torch.Size([2, 200])

    # --- 파라미터 수 확인 ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")  # Swin-T 기준 약 28M
    pass
