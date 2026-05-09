import os
import shutil

def prepare_tiny_imagenet_val(dataset_root):
    """
    Tiny ImageNet의 val 디렉토리를 ImageFolder 형식에 맞게 재구성합니다.
    구조 변경 전: val/images/val_0.jpg, val_1.jpg ...
    구조 변경 후: val/n01443537/val_0.jpg, val/n01443537/val_1.jpg ...
    """
    val_dir = os.path.join(dataset_root, 'val')
    img_dir = os.path.join(val_dir, 'images')
    annot_file = os.path.join(val_dir, 'val_annotations.txt')

    # 1. 어노테이션 파일 읽기
    with open(annot_file, 'r') as f:
        lines = f.readlines()

    # 2. 각 라인별로 이미지 이동 (파일 이름 \t 클래스 아이디 \t ...)
    for line in lines:
        parts = line.strip().split('\t')
        img_name = parts[0]
        class_id = parts[1]

        # 클래스별 폴더 생성
        target_dir = os.path.join(val_dir, class_id)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 이미지 이동 (val/images/xxx.jpg -> val/class_id/xxx.jpg)
        src_path = os.path.join(img_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    # 3. 빈 images 폴더 삭제
    if os.path.exists(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)
        print("Validation set preprocessing complete.")

prepare_tiny_imagenet_val('./data/tiny-imagenet-200')