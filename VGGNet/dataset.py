import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_dataloaders(batch_size=256, num_workers=0, isSingle=False, val_scale=64):
    return get_train_loader(batch_size, num_workers, isSingle), get_val_loader(batch_size, num_workers, isSingle, val_scale)

def get_train_loader(batch_size=256, num_workers=0, isSingle=False):
    train_single_transform = transforms.Compose([
        transforms.Resize((72)),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transform_norm
    ])

    train_multi_trainsfrom = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transform_norm
    ])

    if isSingle:
        train_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=train_single_transform)
    else:
        train_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=train_multi_trainsfrom)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader

def get_val_loader(batch_size=256, num_workers=0, isSingle=False, val_scale=72):
    val_single_transform = transforms.Compose([
        transforms.Resize(val_scale),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transform_norm
    ])
    
    val_multi_transform = transforms.Compose([
        transforms.Resize(val_scale),
        transforms.TenCrop(64),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transform_norm(c) for c in crops])),
    ])

    if isSingle:
        val_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val', transform=val_single_transform)
    else:
        val_dataset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val', transform=val_multi_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return val_loader