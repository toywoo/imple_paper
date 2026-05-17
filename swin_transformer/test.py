"""
=============================================================================
Swin Transformer 테스트 & 추론 가이드
- 학습된 모델 로드
- 테스트 데이터셋 평가
- 단일 이미지 추론
- 결과 시각화
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import swin_t, Swin_T_Weights

# from swin_transformer import SwinTransformer


# =============================================================================
# Step 1: 학습된 모델 로드
# =============================================================================
"""
[방법 1: 직접 구현 모델]
- SwinTransformer(num_classes=200) 생성 → torch.load → load_state_dict

[방법 2: timm 모델]
- timm.create_model(..., num_classes=200) → load_state_dict

⚠️ 체크포인트 dict 구조: {'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc'}
"""

# TODO: 체크포인트 로드 → 모델에 가중치 적용 → model.eval()
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 200)
model = model.to(device)
checkpoint = torch.load("swin_tiny_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])

from train import set_dateset
_, test_loader = set_dateset(batch_size)

# =============================================================================
# Step 2: 테스트 데이터셋 평가
# =============================================================================
@torch.no_grad()
def test(model, test_loader, device):
    """
    [구현 힌트]
    1. model.eval()
    2. Top-1 accuracy: outputs.max(1) → predicted.eq(labels) 비교
    3. Top-5 accuracy: outputs.topk(5, dim=1) 활용
    4. (선택) 클래스별 정확도 추적: dict로 클래스별 correct/total 집계
    5. 가장 잘 맞추는/못 맞추는 클래스 분석하면 모델 이해에 도움
    """
    # TODO: Top-1, Top-5 accuracy 계산
    # TODO: (선택) 클래스별 정확도 분석
    model.eval()  # 평가 모드 (드롭아웃, 배치노름 등이 평가용으로 바뀜)
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    class_correct = torch.zeros(200).to(device)
    class_total = torch.zeros(200).to(device)
    total = 0
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 1. 예측 및 손실 계산 (No Grad 상태    이므로 backward는 안 함)
        outputs = model(inputs)
        
        # 통계 계산
        total += labels.size(0)
        
        # Top-1 Accuracy 계산
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        
        # Top-5 Accuracy 계산 (예측값 중 상위 5개 안에 정답이 있는지 확인)
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()

        # 클래스 별 correct / total
        c = predicted.eq(labels)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    class_acc = 100 * class_correct / (class_total + 1e-6)
    sorted_acc, sorted_idx = torch.sort(class_acc, descending=True)
        
    val_acc_top1 = 100. * correct_top1 / total
    val_acc_top5 = 100. * correct_top5 / total

    print(f"Test | Top-1 Acc: {val_acc_top1:.2f}% | Top-5 Acc: {val_acc_top5:.2f}% |")
    print("\n=== 가장 잘 맞추는 클래스 Top 5 ===")
    for i in range(5):
        print(f"Class {sorted_idx[i].item()}: {sorted_acc[i].item():.2f}%")
        
    print("\n=== 가장 못 맞추는 클래스 Top 5 ===")
    for i in range(1, 6):
        print(f"Class {sorted_idx[-i].item()}: {sorted_acc[-i].item():.2f}%")


# =============================================================================
# Step 3: 단일 이미지 추론
# =============================================================================
def predict_single_image(model, image_path, transform, class_names, device, top_k=5):
    """
    [구현 힌트]
    1. PIL.Image.open → transform → unsqueeze(0) 으로 배치 차원 추가
    2. model(input_tensor) → torch.softmax → topk로 상위 K개 예측
    3. class_names[idx]로 클래스 이름 출력
    """
    # TODO: 이미지 로드 → 전처리 → 추론 → Top-K 결과 출력
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        
        probabilities = torch.softmax(outputs, dim=1)
        prob, idx = probabilities.topk(top_k, dim=1)
        
    print(f"\n=== 이미지: {image_path} 추론 결과 ===")
    for i in range(top_k):
        class_idx = idx[0][i].item()
        probability = prob[0][i].item() * 100
        
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        print(f"Top {i+1}: {class_name} ({probability:.2f}%)")

# =============================================================================
# Step 4: 결과 시각화
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image

def plot_learning_curves(log_path):
    """
    train.log 파일을 파싱하여 학습 곡선을 그립니다.
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
            
        train_matches = re.findall(r"Train \| Loss: ([\d.]+) \| Acc: ([\d.]+)%", log_content)
        val_matches = re.findall(r"Validation \| Loss: ([\d.]+) \| Top-1 Acc: ([\d.]+)%", log_content)
        
        for loss, acc in train_matches:
            train_losses.append(float(loss))
            train_accs.append(float(acc))
            
        for loss, acc in val_matches:
            val_losses.append(float(loss))
            val_accs.append(float(acc))
            
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-', label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        print("학습 곡선이 'learning_curves.png'로 저장되었습니다.")
        plt.show()
        
    except FileNotFoundError:
        print(f"로그 파일 {log_path}를 찾을 수 없습니다.")

def visualize_predictions(model, test_loader, device, class_names=None, num_images=6):
    """
    테스트 데이터셋에서 이미지를 가져와 예측 결과를 시각화합니다.
    """
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(15, 10))
    
    # ImageNet 정규화 역변환용
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 3 + 1, 3, images_so_far)
                ax.axis('off')
                
                # 텐서를 이미지로 변환 (역정규화)
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                true_label = labels[j].item()
                pred_label = preds[j].item()
                
                true_name = class_names[true_label] if class_names else f"Class {true_label}"
                pred_name = class_names[pred_label] if class_names else f"Class {pred_label}"
                
                color = 'green' if true_label == pred_label else 'red'
                
                ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
                plt.imshow(img)
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('predictions_sample.png')
                    print("예측 결과 샘플이 'predictions_sample.png'로 저장되었습니다.")
                    plt.show()
                    return

# =============================================================================
# Step 5: Attention Map 시각화 (Advanced - 선택)
# =============================================================================
def visualize_gradcam(model, image_path, transform, device):
    """
    GradCAM을 이용하여 모델이 어디를 보고 판단하는지 시각화합니다.
    ⚠️ pytorch-grad-cam 라이브러리가 설치되어 있어야 합니다.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        # torchvision SwinTransformer의 마지막 stage의 마지막 block의 norm 레이어 타겟
        # 모델 구조에 따라 다를 수 있습니다.
        target_layers = [model.features[-1][-1].norm2]
        
        cam = GradCAM(model=model, target_layers=target_layers)
        
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # 원본 이미지 (역정규화 안 하고 그냥 원본 쓰면 편함)
        img_np = np.array(img.resize((224, 224))) / 255.0
        
        # 예측 클래스 추출
        outputs = model(input_tensor)
        target_category = torch.argmax(outputs, dim=1).item()
        
        targets = [ClassifierOutputTarget(target_category)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img_np)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f'GradCAM (Class {target_category})')
        plt.imshow(visualization)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('gradcam_result.png')
        print("GradCAM 결과가 'gradcam_result.png'로 저장되었습니다.")
        plt.show()
        
    except ImportError:
        print("GradCAM을 사용하려면 'pip install pytorch-grad-cam'이 필요합니다.")
        print("라이브러리가 없으므로 시각화를 건너뜁니다.")

if __name__ == "__main__":
    print("=== 1. 테스트 데이터셋 평가 시작 ===")
    test(model, test_loader, device)
    
    print("\n=== 2. 학습 곡선 시각화 ===")
    plot_learning_curves('train.log')
    
    print("\n=== 3. 예측 결과 샘플 시각화 ===")
    visualize_predictions(model, test_loader, device, class_names=None, num_images=6)
    
    print("\n=== 4. 단일 이미지 추론 및 GradCAM ===")
    # ⚠️ 테스트해 볼 실제 이미지 경로를 하나 적어주셔야 합니다!
    sample_image_path = "./dataset/tiny-imagenet-200/val/n01443537/val_1230.JPEG" 
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # GradCAM 시각화 (gradcam_result.png 저장)
    visualize_gradcam(model, sample_image_path, val_transform, device)
    
    try:
        # 단일 이미지 추론
        predict_single_image(model, sample_image_path, val_transform, class_names=None, device=device)
    except FileNotFoundError:
        print(f"\n[안내] 테스트용 이미지({sample_image_path})를 찾을 수 없어 단일 이미지 추론은 건너뜁니다.")

