from torchvision.transforms.functional import center_crop
import torch, torch.nn as nn

class UEncBlock(nn.Module):
    """
    U-Net의 Encoder 블록
    Conv + ReLU #2, MaxPool으로 다운샘플링

    Args:
        in_channels  (int): 입력 채널 수
        out_channels (int): 출력 채널 수 (첫 번째 conv 이후 유지)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feat = None

    """
    Args:    x (Tensor): (B, in_channels, H, W)
    Returns: x (Tensor): (B, out_channels, H/2, W/2)
    """
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        self.feat = x.clone().detach()
        x = self.maxpool(x)
        return x

class UDecBlock(nn.Module):
    """
    U-Net의 Decoder 블록
    Conv + ReLU #2, Conv 2x2로 업샘플링

    Args:
        in_channels  (int): 입력 채널 수
        out_channels (int): 첫번째 conv 출력 채널 수
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()
        # up_conv: 업샘플링을 수행하는 Transposed Convolution
        # 1. (out_channels // 2): 다음 층의 Skip Connection(Concat) 후의 채널 수를 맞추기 위해 채널을 절반으로 줄임
        # 2. stride=2: 공간 해상도(H, W)를 2배로 키움 (Upsampling)
        self.up_conv = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    """
        Args:    x (Tensor): (B, in_channels, H, W)
        Returns: x (Tensor): (B, out_channels // 2, H/2, W/2)
    """
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.up_conv(x)
        return x

class UOutBlock(nn.Module):
    """
    U-Net의 출력 변환 블록
    Conv + ReLU #2, Conv 1x1로 분류할 클래스 수로 변환한다.

    Args:
        in_channels  (int): 입력 채널 수
        out_channels (int): 첫번째 conv 출력 채널 수
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(out_channels, 2, kernel_size=1)
        self.relu = nn.ReLU()

    """
    Args:    x (Tensor): (B, in_channels, H, W)
    Returns: x (Tensor): (B, 2, H, W)
    """
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class UNet(nn.Module):
    """
    U-Net Biomedical Image 분할 알고리즘
    UEncBlock(feature extractor) #4 + UDecBlock(upsampler) #4 + UOutBlock로 구성됨


    """
    def __init__(self):
        super().__init__()
        self.UEncBlock64 = UEncBlock(1, 64)
        self.UEncBlock128 = UEncBlock(64, 128)
        self.UEncBlock256 = UEncBlock(128, 256)
        self.UEncBlock512 = UEncBlock(256, 512)
        self.UDecBlock1024 = UDecBlock(512, 1024)
        self.UDecBlock512 = UDecBlock(1024, 512)
        self.UDecBlock256 = UDecBlock(512, 256)
        self.UDecBlock128 = UDecBlock(256, 128)
        self.UOutBlock = UOutBlock(128, 64)

    """
    Args:    x (Tensor): (B, 1, 572, 572)
    Returns: y (Tensor): (B, 2, 388, 388)
    """
    def forward(self, x):
        x = self.UEncBlock64(x)
        x = self.UEncBlock128(x)
        x = self.UEncBlock256(x)
        x = self.UEncBlock512(x)
        x = self.UDecBlock1024(x)
        
        c1024 = self.concat_crop(self.UEncBlock512.feat, x)
        y = self.UDecBlock512(c1024)
        c512 = self.concat_crop(self.UEncBlock256.feat, y)
        y = self.UDecBlock256(c512)
        c256 = self.concat_crop(self.UEncBlock128.feat, y)
        y = self.UDecBlock128(c256)
        c64 = self.concat_crop(self.UEncBlock64.feat, y)
        y = self.UOutBlock(c64)

        return y

    """
    Args:    enc_feat, dec_feat (Tensor, Tensor): (B, C/2, enc_H, enc_W), (B, C/2, dec_H, dec_W)
    Returns: cat_feat (Tensor): (B, C, dec_H, dec_W)
    """
    def concat_crop(self, enc_feat, dec_feat):
        crop_enc_feat = center_crop(enc_feat, [dec_feat.size(2), dec_feat.size(3)])
        cat_feat = torch.cat([crop_enc_feat, dec_feat], dim=1)
        return cat_feat


if __name__ == '__main__':
    x = torch.rand(3, 1, 572, 572) # batch가 3인 이미지 텐서 (3, 1, 572, 572) 생성
    model = UNet() # 네트워크 객체 생성
    out = model(x)
    print(f"Output shape: {out.shape}") # 출력 형식 확인(3, 2, 388, 388)