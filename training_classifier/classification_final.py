import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from typing import Dict
import random


class BirdClassifier(nn.Module):

    def __init__(self, num_classes, use_pretrained=True):
        super(BirdClassifier, self).__init__()

        if use_pretrained:
            weights_path = os.path.join(os.path.dirname(__file__), 'mobilenet_v2-b0353104.pth')

            if os.path.exists(weights_path):
                mobilenet = models.mobilenet_v2(weights=None)
                state_dict = torch.load(weights_path, map_location='cpu')
                mobilenet.load_state_dict(state_dict)
            else:
                try:
                    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                except:
                    try:
                        mobilenet = models.mobilenet_v2(pretrained=True)
                    except:
                        mobilenet = models.mobilenet_v2(weights=None)
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        self.features = mobilenet.features

        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.features[-4:].parameters():
            param.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class BirdDataset(Dataset):

    def __init__(self, img_classes, img_dir, transform=None):
        self.img_classes = img_classes
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = list(img_classes.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        label = self.img_classes[img_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_classifier(train_gt: Dict[str, int], train_img_dir: str, fast_train: bool = False):
    num_classes = max(train_gt.values()) + 1
    print(f"Number of classes: {num_classes}")
    print(f"Number of training images: {len(train_gt)}")

    device = torch.device('cpu')

    use_pretrained = not fast_train
    model = BirdClassifier(num_classes, use_pretrained=use_pretrained).to(device)

    if fast_train:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        num_epochs = 1
        max_batches = 5
        batch_size = 32
        lr = 0.001
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        num_epochs = 25
        max_batches = None
        batch_size = 32
        lr = 0.0001

    dataset = BirdDataset(train_gt, train_img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    if not fast_train:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        num_batches = min(len(loader), max_batches or len(loader))
        avg_loss = running_loss / num_batches
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if not fast_train:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if old_lr != new_lr:
                print(f'Learning rate reduced from {old_lr} to {new_lr}')

            if avg_loss < best_loss:
                best_loss = avg_loss

    return model


def classify(model_path: str, test_img_dir: str) -> Dict[str, int]:
    device = torch.device('cpu')

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get('num_classes', None)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        num_classes = None

    if num_classes is None:
        num_classes = state_dict['classifier.weight'].shape[0]

    model = BirdClassifier(num_classes, use_pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(test_img_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    results = {}

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(test_img_dir, img_name)

            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)

                output = model(image)
                _, predicted = torch.max(output, 1)
                results[img_name] = predicted.item()

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                results[img_name] = 0

    return results


if __name__ == '__main__':
    import sys
    from os.path import join, exists, abspath


    def read_csv(filename):
        if not exists(filename):
            raise FileNotFoundError(f"File not found: {abspath(filename)}")

        res = {}
        with open(filename) as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    fname, class_id = parts
                    res[fname] = int(class_id)
        return res


    if len(sys.argv) < 2:
        print("Usage: python classification.py <path_to_train_dir>")
        print("\nDownload pretrained weights:")
        print("wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
        sys.exit(1)

    train_dir = sys.argv[1]

    if not exists(train_dir):
        print(f"Error: Directory not found: {abspath(train_dir)}")
        sys.exit(1)

    gt_path = join(train_dir, 'gt.csv')
    img_dir = join(train_dir, 'images')

    train_gt = read_csv(gt_path)

    print("Starting training...")
    model = train_classifier(train_gt, img_dir, fast_train=False)

    print("\nSaving model to birds_model.pt...")
    num_classes = max(train_gt.values()) + 1
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes
    }, 'birds_model.pt')

    print("Done! Model saved successfully.")