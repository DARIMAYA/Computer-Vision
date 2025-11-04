# detection_fast.py - Быстрая версия для тестирования
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from typing import Dict


class FastFacePointsDataset(Dataset):
    def __init__(self, gt_dict: Dict, img_dir: str, target_size=(128, 128)):
        self.gt_dict = gt_dict
        self.img_dir = img_dir
        self.target_size = target_size
        self.filenames = list(gt_dict.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        original_width, original_height = image.size
        keypoints = self.gt_dict[filename].copy().reshape(-1, 2)

        # Ресайз
        image_resized = image.resize(self.target_size, Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Нормализация координат к [0, 1]
        keypoints[:, 0] = keypoints[:, 0] / original_width
        keypoints[:, 1] = keypoints[:, 1] / original_height

        return image_tensor, torch.FloatTensor(keypoints.flatten()), filename


class FastFacePointsModel(nn.Module):
    def __init__(self, num_points=28):
        super(FastFacePointsModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_points)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


def train_detector(train_gt: Dict, train_img_dir: str, fast_train=True):
    if fast_train:
        epochs = 2
        batch_size = 16
        target_size = (128, 128)
        device = 'cpu'
    else:
        epochs = 50
        batch_size = 32
        target_size = (128, 128)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Fast training: {fast_train}, Epochs: {epochs}")

    dataset = FastFacePointsDataset(train_gt, train_img_dir, target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = FastFacePointsModel(num_points=28)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        for images, keypoints, _ in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            outputs = model(images)
            loss = criterion(outputs, keypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            if fast_train and batch_count >= 2:
                break

        avg_loss = running_loss / batch_count
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}')

        if fast_train:
            break

    if not fast_train:
        torch.save(model.state_dict(), 'facepoints_model.pt')
        print('✓ Model saved')

    return model


def detect(model_path: str, test_img_dir: str) -> Dict[str, np.ndarray]:
    device = 'cpu'
    target_size = (128, 128)

    model = FastFacePointsModel(num_points=28)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        test_files = [f for f in os.listdir(test_img_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return {f: np.zeros(28) for f in test_files}

    model.to(device)
    model.eval()

    test_files = [f for f in os.listdir(test_img_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = {}

    with torch.no_grad():
        for filename in test_files:
            img_path = os.path.join(test_img_dir, filename)

            try:
                image = Image.open(img_path)
                original_width, original_height = image.size

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image_resized = image.resize(target_size, Image.BILINEAR)
                image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0
                image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                pred_keypoints = model(image_tensor).cpu().numpy()[0]
                pred_keypoints = pred_keypoints.reshape(-1, 2)

                pred_keypoints[:, 0] = pred_keypoints[:, 0] * original_width
                pred_keypoints[:, 1] = pred_keypoints[:, 1] * original_height

                results[filename] = pred_keypoints.flatten()

            except Exception as e:
                results[filename] = np.zeros(28)

    return results


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


# Для быстрого тестирования
if __name__ == "__main__":
    train_gt = read_csv('tests/00_test_img_input/train/gt.csv')
    train_img_dir = 'tests/00_test_img_input/train/images'

    print("Fast training with 50 epochs...")
    model = train_detector(train_gt, train_img_dir, fast_train=False)