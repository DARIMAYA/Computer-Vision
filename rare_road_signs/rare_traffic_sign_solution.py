import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetRTSD(torch.utils.data.Dataset):
    def __init__(
            self,
            root_folders: typing.List[str],
            path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        self.samples = []
        for root_folder in root_folders:
            if not os.path.exists(root_folder):
                continue
            for class_name in os.listdir(root_folder):
                class_path = f"{root_folder}/{class_name}"
                if not os.path.isdir(class_path):
                    continue
                if class_name not in self.class_to_idx:
                    continue
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_path):
                    img_path = f"{class_path}/{img_name}"
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, class_idx))

        self.classes_to_samples = {i: [] for i in range(len(self.classes))}
        for idx, (_, class_idx) in enumerate(self.samples):
            self.classes_to_samples[class_idx].append(idx)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.3),
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                rotate=(-10, 10),
                p=0.3
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        img_path, class_idx = self.samples[index]
        image = skimage.io.imread(img_path)

        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        image = self.transform(image=image)["image"]
        return image, img_path, class_idx

    @staticmethod
    def get_classes(
            path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        with open(path_to_classes_json, 'r') as f:
            classes_info = json.load(f)

        class_to_idx = {}
        classes = [None] * len(classes_info)

        for class_name, class_data in classes_info.items():
            idx = class_data['id']
            class_to_idx[class_name] = idx
            classes[idx] = class_name

        return classes, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            path_to_classes_json: str,
            annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root

        self.samples = []
        for img_name in sorted(os.listdir(root)):
            img_path = os.path.join(root, img_name)
            if os.path.isfile(img_path):
                self.samples.append(img_name)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.targets = None
        if annotations_file is not None:
            _, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
            self.targets = {}
            with open(annotations_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['filename']
                    class_name = row['class']
                    self.targets[filename] = class_to_idx[class_name]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        filename = self.samples[index]
        img_path = os.path.join(self.root, filename)
        image = skimage.io.imread(img_path)

        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        image = self.transform(image=image)["image"]

        class_idx = -1
        if self.targets is not None:
            class_idx = self.targets[filename]

        return image, filename, class_idx

    def __len__(self) -> int:
        return len(self.samples)


class CustomNetwork(L.LightningModule):
    def __init__(
            self,
            features_criterion: (
                    typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
            ) = None,
            internal_features: int = 1024,
    ):
        super().__init__()

        resnet = torchvision.models.resnet50(weights=None)

        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])

        resnet_out_features = 2048
        self.fc1 = torch.nn.Linear(resnet_out_features, internal_features)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(internal_features, CLASSES_CNT)

        self.features_criterion = features_criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        features = self.fc1(x)
        features = self.relu(features)

        logits = self.fc2(features)

        return features, logits

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy()

    def training_step(self, batch, batch_idx):
        images, _, labels = batch
        features, logits = self.forward(images)

        loss = self.criterion(logits, labels)

        if self.features_criterion is not None:
            features_loss = self.features_criterion(features, labels)
            loss = loss + features_loss

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def train_simple_classifier() -> torch.nn.Module:
    print("\n" + "=" * 50)
    print("TRAINING SIMPLE CLASSIFIER")
    print("=" * 50)

    train_dataset = DatasetRTSD(
        root_folders=["cropped-train"],
        path_to_classes_json="classes.json"
    )

    print(f"Dataset size: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    model = CustomNetwork()

    trainer = L.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader)

    torch.save(model.state_dict(), "simple_model.pt")
    print("✓ Model saved to simple_model.pt")

    return model


def apply_classifier(
        model: torch.nn.Module,
        test_folder: str,
        path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    test_dataset = TestData(
        root=test_folder,
        path_to_classes_json=path_to_classes_json
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model.eval()
    model.to(DEVICE)

    results = []
    classes, _ = DatasetRTSD.get_classes(path_to_classes_json)

    with torch.no_grad():
        for images, filenames, _ in test_loader:
            images = images.to(DEVICE)
            predictions = model.predict(images)

            for filename, pred_idx in zip(filenames, predictions):
                results.append({
                    'filename': filename,
                    'class': classes[pred_idx]
                })

    return results


def test_classifier(
        model: torch.nn.Module,
        test_folder: str,
        annotations_file: str,
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    predictions = apply_classifier(model, test_folder, "classes.json")

    with open("classes.json", 'r') as f:
        classes_info = json.load(f)

    gt = {}
    with open(annotations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['filename']] = row['class']

    total_correct = 0
    total_count = 0
    rare_correct = 0
    rare_count = 0
    freq_correct = 0
    freq_count = 0

    for pred in predictions:
        filename = pred['filename']
        pred_class = pred['class']
        true_class = gt[filename]

        is_correct = (pred_class == true_class)
        class_type = classes_info[true_class]['type']

        total_count += 1
        if is_correct:
            total_correct += 1

        if class_type == 'rare':
            rare_count += 1
            if is_correct:
                rare_correct += 1
        else:
            freq_count += 1
            if is_correct:
                freq_correct += 1

    total_acc = total_correct / total_count if total_count > 0 else 0
    rare_recall = rare_correct / rare_count if rare_count > 0 else 0
    freq_recall = freq_correct / freq_count if freq_count > 0 else 0

    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    def __init__(self, background_path: str) -> None:
        super().__init__()
        self.background_path = background_path
        self.background_files = [
            os.path.join(background_path, f)
            for f in os.listdir(background_path)
            if os.path.isfile(os.path.join(background_path, f))
        ]

    @staticmethod
    def resize_icon(icon: np.ndarray, size: int) -> np.ndarray:
        return skimage.transform.resize(
            icon, (size, size), anti_aliasing=True, preserve_range=True
        ).astype(np.uint8)

    @staticmethod
    def add_padding(icon: np.ndarray, padding_percent: float) -> np.ndarray:
        h, w = icon.shape[:2]
        pad = int(max(h, w) * padding_percent)

        if len(icon.shape) == 3:
            channels = icon.shape[2]
            padded = np.zeros((h + 2 * pad, w + 2 * pad, channels), dtype=icon.dtype)
            padded[pad:pad + h, pad:pad + w] = icon
        else:
            padded = np.pad(icon, ((pad, pad), (pad, pad)), mode='constant')

        return padded

    @staticmethod
    def normalize_channels(icon: np.ndarray) -> np.ndarray:
        if len(icon.shape) == 2:
            icon = np.stack([icon, icon, icon], axis=-1)
        elif icon.shape[2] == 2:
            icon = np.stack([icon[:, :, 0], icon[:, :, 0], icon[:, :, 0]], axis=-1)
        elif icon.shape[2] == 1:
            icon = np.stack([icon[:, :, 0], icon[:, :, 0], icon[:, :, 0]], axis=-1)
        return icon

    @staticmethod
    def change_color_hsv(icon: np.ndarray) -> np.ndarray:
        icon = SignGenerator.normalize_channels(icon)

        has_alpha = icon.shape[2] == 4
        if has_alpha:
            alpha = icon[:, :, 3].copy()

        rgb = icon[:, :, :3].astype(np.float32) / 255.0
        hsv = skimage.color.rgb2hsv(rgb)

        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.uniform(-0.05, 0.05), 0, 1)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.9, 1.1), 0, 1)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.9, 1.1), 0, 1)

        rgb = skimage.color.hsv2rgb(hsv)
        rgb = (rgb * 255).astype(np.uint8)

        if has_alpha:
            result = np.dstack([rgb, alpha])
        else:
            result = rgb

        return result

    @staticmethod
    def rotate_icon(icon: np.ndarray, angle: float) -> np.ndarray:
        rotated = skimage.transform.rotate(
            icon, angle, resize=True, preserve_range=True
        ).astype(np.uint8)

        if len(rotated.shape) == 2:
            rotated = np.stack([rotated, rotated, rotated], axis=-1)
        elif rotated.shape[2] < 3:
            rotated = SignGenerator.normalize_channels(rotated)

        return rotated

    @staticmethod
    def motion_blur(icon: np.ndarray) -> np.ndarray:
        icon = SignGenerator.normalize_channels(icon)

        size = random.randint(3, 7)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0
        kernel = kernel / size

        angle = random.uniform(-90, 90)
        kernel = skimage.transform.rotate(kernel, angle, preserve_range=True)
        kernel = kernel / (kernel.sum() + 1e-10)

        result = np.zeros_like(icon)
        for i in range(icon.shape[2]):
            result[:, :, i] = scipy.ndimage.convolve(
                icon[:, :, i].astype(float), kernel, mode='constant'
            )

        return result.astype(np.uint8)

    @staticmethod
    def gaussian_blur(icon: np.ndarray) -> np.ndarray:
        icon = SignGenerator.normalize_channels(icon)

        sigma = random.uniform(0, 1.0)
        result = np.zeros_like(icon)
        for i in range(icon.shape[2]):
            result[:, :, i] = skimage.filters.gaussian(
                icon[:, :, i], sigma=sigma, preserve_range=True
            )
        return result.astype(np.uint8)

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        icon = self.normalize_channels(icon)

        size = random.randint(16, 128)
        icon = self.resize_icon(icon, size)

        padding_percent = random.uniform(0, 0.15)
        icon = self.add_padding(icon, padding_percent)

        icon = self.change_color_hsv(icon)

        angle = random.uniform(-15, 15)
        icon = self.rotate_icon(icon, angle)

        if random.random() < 0.3:
            icon = self.motion_blur(icon)

        icon = self.gaussian_blur(icon)

        bg_file = random.choice(self.background_files)
        bg = skimage.io.imread(bg_file)

        if len(bg.shape) == 2:
            bg = np.stack([bg, bg, bg], axis=-1)
        elif bg.shape[2] == 4:
            bg = bg[:, :, :3]

        h, w = icon.shape[:2]
        bg_h, bg_w = bg.shape[:2]

        if bg_h < h or bg_w < w:
            scale = max(h / bg_h, w / bg_w) * 1.2
            new_h, new_w = int(bg_h * scale), int(bg_w * scale)
            bg = skimage.transform.resize(
                bg, (new_h, new_w), anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)
            bg_h, bg_w = bg.shape[:2]

        y = random.randint(0, max(0, bg_h - h))
        x = random.randint(0, max(0, bg_w - w))

        bg_crop = bg[y:y + h, x:x + w].copy()

        if bg_crop.shape[:2] != icon.shape[:2]:
            bg_crop = skimage.transform.resize(
                bg_crop, icon.shape[:2], anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)

        if icon.shape[2] == 4:
            mask = icon[:, :, 3:4].astype(float) / 255.0
            icon_rgb = icon[:, :, :3]
            result = (icon_rgb * mask + bg_crop * (1 - mask)).astype(np.uint8)
        else:
            result = icon[:, :, :3]

        return result


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    icon_path, output_folder, background_path, samples_per_class = args

    icon = skimage.io.imread(icon_path)
    class_name = os.path.splitext(os.path.basename(icon_path))[0]

    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

    generator = SignGenerator(background_path)

    for i in range(samples_per_class):
        sample = generator.get_sample(icon.copy())
        output_path = os.path.join(class_folder, f"{i}.png")
        skimage.io.imsave(output_path, sample)


def generate_all_data(
        output_folder: str,
        icons_path: str,
        background_path: str,
        samples_per_class: int = 1000,
) -> None:
    print("\n" + "=" * 50)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 50)
    print(f"Output folder: {output_folder}")
    print(f"Samples per class: {samples_per_class}")

    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params), total=len(params)))

    print("✓ Synthetic data generated")


def train_synt_classifier() -> torch.nn.Module:
    print("\n" + "=" * 50)
    print("TRAINING CLASSIFIER WITH SYNTHETIC DATA")
    print("=" * 50)

    train_dataset = DatasetRTSD(
        root_folders=["cropped-train", "synthetic_signs"],
        path_to_classes_json="classes.json"
    )

    print(f"Dataset size: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    model = CustomNetwork()

    trainer = L.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader)

    torch.save(model.state_dict(), "simple_model_with_synt.pt")
    print("✓ Model saved to simple_model_with_synt.pt")

    return model


class FeaturesLoss(torch.nn.Module):
    def __init__(self, margin: float = 2.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = outputs.size(0)

        distances = torch.cdist(outputs, outputs, p=2)

        labels_expand = labels.unsqueeze(1)
        same_class = (labels_expand == labels_expand.t()).float()
        diff_class = 1 - same_class

        mask = 1 - torch.eye(batch_size, device=outputs.device)
        same_class = same_class * mask
        diff_class = diff_class * mask

        same_loss = same_class * distances ** 2
        diff_loss = diff_class * torch.clamp(self.margin - distances, min=0) ** 2

        same_count = same_class.sum()
        diff_count = diff_class.sum()

        loss = 0
        if same_count > 0:
            loss += same_loss.sum() / (2 * same_count)
        if diff_count > 0:
            loss += diff_loss.sum() / (2 * diff_count)

        return loss


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    def __init__(
            self,
            data_source: DatasetRTSD,
            elems_per_class: int,
            classes_per_batch: int,
    ) -> None:
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.classes_to_samples = data_source.classes_to_samples

        self.available_classes = [
            class_idx for class_idx, samples in self.classes_to_samples.items()
            if len(samples) > 0
        ]

        self.batch_size = classes_per_batch * elems_per_class
        self.num_batches = len(data_source) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_classes = random.sample(
                self.available_classes,
                min(self.classes_per_batch, len(self.available_classes))
            )

            batch = []
            for class_idx in selected_classes:
                samples = self.classes_to_samples[class_idx]
                selected_samples = random.choices(samples, k=self.elems_per_class)
                batch.extend(selected_samples)

            yield batch

    def __len__(self) -> int:
        return self.num_batches


def train_better_model() -> torch.nn.Module:
    print("\n" + "=" * 50)
    print("TRAINING IMPROVED MODEL WITH FEATURES LOSS")
    print("=" * 50)

    train_dataset = DatasetRTSD(
        root_folders=["cropped-train", "synthetic_signs"],
        path_to_classes_json="classes.json"
    )

    print(f"Dataset size: {len(train_dataset)}")

    batch_sampler = CustomBatchSampler(
        train_dataset,
        elems_per_class=4,
        classes_per_batch=8
    )

    num_workers = 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        persistent_workers=False
    )

    features_criterion = FeaturesLoss(margin=2.0)
    model = CustomNetwork(features_criterion=features_criterion)

    trainer = L.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader)

    torch.save(model.state_dict(), "improved_features_model.pt")
    print("✓ Model saved to improved_features_model.pt")

    return model


class ModelWithHead(CustomNetwork):
    def __init__(self, n_neighbors: int = 10) -> None:
        super().__init__()
        self.eval()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
        self.n_neighbors = n_neighbors

    def load_nn(self, nn_weights_path: str) -> None:
        state_dict = torch.load(
            nn_weights_path,
            map_location=DEVICE,
            weights_only=False
        )
        self.load_state_dict(state_dict)
        self.eval()
        self.to(DEVICE)

    def load_head(self, knn_path: str) -> None:
        with open(knn_path, 'rb') as f:
            self.knn = pickle.load(f)

    def save_head(self, knn_path: str) -> None:
        with open(knn_path, 'wb') as f:
            pickle.dump(self.knn, f)

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        self.eval()
        self.to(DEVICE)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, _, labels in indexloader:
                images = images.to(DEVICE)
                features, _ = self.forward(images)

                features = features.cpu().numpy()
                features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

                all_features.append(features)
                all_labels.append(labels.numpy())

        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)

        self.knn.fit(all_features, all_labels)

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            features, model_pred = self.forward(imgs)
            features = features.cpu().numpy()
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
            knn_pred = self.knn.predict(features)
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        self.data_source = data_source
        self.examples_per_class = examples_per_class
        self.classes_to_samples = data_source.classes_to_samples

        self.indices = []
        for class_idx, samples in self.classes_to_samples.items():
            if len(samples) > 0:
                selected = random.sample(samples, min(examples_per_class, len(samples)))
                while len(selected) < examples_per_class:
                    selected.append(random.choice(samples))
                self.indices.extend(selected)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    print("\n" + "=" * 50)
    print("TRAINING kNN HEAD")
    print("=" * 50)

    index_dataset = DatasetRTSD(
        root_folders=["synthetic_signs"],
        path_to_classes_json="classes.json"
    )

    index_sampler = IndexSampler(index_dataset, examples_per_class)

    print(f"Index size: {len(index_sampler)}")

    index_loader = torch.utils.data.DataLoader(
        index_dataset,
        batch_size=64,
        sampler=index_sampler,
        num_workers=0
    )

    model = ModelWithHead(n_neighbors=5)
    model.load_nn(nn_weights_path)

    model.train_head(index_loader)

    model.save_head("knn_model.pickle")
    print("✓ kNN model saved to knn_model.pickle")

    return model


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("RARE TRAFFIC SIGNS CLASSIFICATION - FULL PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("=" * 70)

    QUICK_MODE = "--quick" in sys.argv
    SKIP_SIMPLE = "--skip-simple" in sys.argv
    SKIP_SYNTH_GEN = "--skip-synth-gen" in sys.argv
    SKIP_SYNTH_TRAIN = "--skip-synth-train" in sys.argv
    SKIP_IMPROVED = "--skip-improved" in sys.argv
    SKIP_KNN = "--skip-knn" in sys.argv

    if QUICK_MODE:
        print("\n⚡ QUICK MODE - Using reduced parameters for testing")
        SAMPLES_PER_CLASS = 100
    else:
        SAMPLES_PER_CLASS = 1000

    if not SKIP_SIMPLE:
        if not os.path.exists("simple_model.pt"):
            train_simple_classifier()
        else:
            print("\n✓ simple_model.pt already exists, skipping...")

        if os.path.exists("smalltest") and os.path.exists("smalltest_annotations.csv"):
            print("\nTesting simple model...")
            model = CustomNetwork()
            model.load_state_dict(torch.load("simple_model.pt"))
            acc, rare, freq = test_classifier(model, "smalltest", "smalltest_annotations.csv")
            print(f"Simple Model - Accuracy: {acc:.4f}, Rare Recall: {rare:.4f}, Freq Recall: {freq:.4f}")

    if not SKIP_SYNTH_GEN:
        if not os.path.exists("synthetic_signs"):
            generate_all_data(
                output_folder="synthetic_signs",
                icons_path="icons",
                background_path="background_images",
                samples_per_class=SAMPLES_PER_CLASS
            )
        else:
            print("\n✓ synthetic_signs folder already exists, skipping generation...")

    if not SKIP_SYNTH_TRAIN:
        if not os.path.exists("simple_model_with_synt.pt"):
            train_synt_classifier()
        else:
            print("\n✓ simple_model_with_synt.pt already exists, skipping...")

        if os.path.exists("smalltest") and os.path.exists("smalltest_annotations.csv"):
            print("\nTesting model with synthetic data...")
            model = CustomNetwork()
            model.load_state_dict(torch.load("simple_model_with_synt.pt"))
            acc, rare, freq = test_classifier(model, "smalltest", "smalltest_annotations.csv")
            print(f"Model with Synth - Accuracy: {acc:.4f}, Rare Recall: {rare:.4f}, Freq Recall: {freq:.4f}")

    if not SKIP_IMPROVED:
        if not os.path.exists("improved_features_model.pt"):
            train_better_model()
        else:
            print("\n✓ improved_features_model.pt already exists, skipping...")

    if not SKIP_KNN:
        if not os.path.exists("knn_model.pickle"):
            train_head("improved_features_model.pt", examples_per_class=20)
        else:
            print("\n✓ knn_model.pickle already exists, skipping...")

        if os.path.exists("smalltest") and os.path.exists("smalltest_annotations.csv"):
            print("\nTesting final model with kNN...")
            model = ModelWithHead(n_neighbors=5)
            model.load_nn("improved_features_model.pt")
            model.load_head("knn_model.pickle")
            acc, rare, freq = test_classifier(model, "smalltest", "smalltest_annotations.csv")
            print(f"Final Model - Accuracy: {acc:.4f}, Rare Recall: {rare:.4f}, Freq Recall: {freq:.4f}")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    if os.path.exists("simple_model.pt"):
        print("  ✓ simple_model.pt")
    if os.path.exists("simple_model_with_synt.pt"):
        print("  ✓ simple_model_with_synt.pt")
    if os.path.exists("improved_features_model.pt"):
        print("  ✓ improved_features_model.pt")
    if os.path.exists("knn_model.pickle"):
        print("  ✓ knn_model.pickle")
    if os.path.exists("synthetic_signs"):
        print("  ✓ synthetic_signs/ folder")

    print("\nYou can now:")
    print("  1. Run unit tests: ./run.py unittest <test_name>")
    print("  2. Submit to the grading system")
    print("\nUsage options:")
    print("  python rare_traffic_sign_solution.py              # Full pipeline")
    print("  python rare_traffic_sign_solution.py --quick      # Quick mode for testing")
    print("  python rare_traffic_sign_solution.py --skip-simple --skip-synth-gen  # Skip specific steps")