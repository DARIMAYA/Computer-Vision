import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from typing import Dict, List


class AvianSpeciesClassifier(nn.Module):
    """
    Классификатор видов птиц с использованием трансферного обучения
    """

    def __init__(self, num_species: int, pretrained_weights: bool = True):
        super(AvianSpeciesClassifier, self).__init__()

        # Инициализация базовой модели
        base_model = self._initialize_base_model(pretrained_weights)

        # Используем только feature extractor
        self.feature_extractor = base_model.features

        # Стратегия заморозки весов
        self._apply_layer_freezing()

        # Слои классификации
        self.spatial_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.regularization_layer = nn.Dropout(p=0.25)
        self.species_predictor = nn.Linear(1280, num_species)

    def _initialize_base_model(self, use_pretrained: bool):
        """Инициализация базовой модели MobileNetV2"""
        if use_pretrained:
            try:
                return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            except Exception:
                return models.mobilenet_v2(weights=None)
        else:
            return models.mobilenet_v2(weights=None)

    def _apply_layer_freezing(self):
        """Заморозка и разморозка слоев для трансферного обучения"""
        # Замораживаем все слои
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Размораживаем последние 4 блока для тонкой настройки
        for parameter in self.feature_extractor[-4:].parameters():
            parameter.requires_grad = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Прямой проход через сеть"""
        features = self.feature_extractor(input_tensor)
        pooled = self.spatial_pooling(features)
        flattened = torch.flatten(pooled, start_dim=1)
        regularized = self.regularization_layer(flattened)
        output = self.species_predictor(regularized)
        return output


class OrnithologyDataset(Dataset):
    """
    Датасет для работы с изображениями птиц
    """

    def __init__(self, species_labels: Dict[str, int],
                 image_directory: str,
                 image_transformations=None):
        self.species_labels = species_labels
        self.image_directory = image_directory
        self.image_transformations = image_transformations
        self.image_filenames = list(species_labels.keys())

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int):
        filename = self.image_filenames[index]
        full_image_path = os.path.join(self.image_directory, filename)

        # Загрузка изображения с обработкой ошибок
        try:
            image_data = Image.open(full_image_path).convert('RGB')
        except Exception as load_error:
            print(f"Ошибка загрузки {full_image_path}: {load_error}")
            image_data = Image.new('RGB', (224, 224), color='gray')

        species_id = self.species_labels[filename]

        if self.image_transformations:
            image_data = self.image_transformations(image_data)

        return image_data, species_id


def train_avian_classifier(training_labels: Dict[str, int],
                           training_images_dir: str,
                           quick_training: bool = False):
    """
    Обучение классификатора видов птиц

    Args:
        training_labels: Словарь с метками изображений
        training_images_dir: Директория с обучающими изображениями
        quick_training: Флаг быстрого обучения для тестирования
    """

    num_species = max(training_labels.values()) + 1
    print(f"Количество распознаваемых видов: {num_species}")
    print(f"Объем обучающей выборки: {len(training_labels)} изображений")

    computation_device = torch.device('cpu')

    # Создание модели
    classifier_model = AvianSpeciesClassifier(num_species,
                                              pretrained_weights=not quick_training)
    classifier_model.to(computation_device)

    # Конфигурация трансформаций изображений
    if quick_training:
        image_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        training_config = {
            'epochs': 1,
            'max_batches': 5,
            'batch_size': 16,
            'learning_rate': 0.0015
        }
    else:
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.15, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        training_config = {
            'epochs': 20,
            'max_batches': None,
            'batch_size': 24,
            'learning_rate': 0.00012
        }

    # Подготовка данных
    training_dataset = OrnithologyDataset(training_labels,
                                          training_images_dir,
                                          image_transforms)
    data_loader = DataLoader(training_dataset,
                             batch_size=training_config['batch_size'],
                             shuffle=True,
                             num_workers=0)

    # Настройка обучения
    loss_criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad,
                              classifier_model.parameters())
    model_optimizer = optim.Adam(trainable_params,
                                 lr=training_config['learning_rate'],
                                 weight_decay=1.1e-4)

    if not quick_training:
        learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.6, patience=4)

    # Цикл обучения
    for current_epoch in range(training_config['epochs']):
        classifier_model.train()
        total_epoch_loss = 0.0
        correct_predictions = 0
        processed_samples = 0

        for batch_index, (batch_images, batch_labels) in enumerate(data_loader):
            if training_config['max_batches'] and batch_index >= training_config['max_batches']:
                break

            batch_images = batch_images.to(computation_device)
            batch_labels = batch_labels.to(computation_device)

            model_optimizer.zero_grad()
            model_output = classifier_model(batch_images)
            batch_loss = loss_criterion(model_output, batch_labels)
            batch_loss.backward()
            model_optimizer.step()

            total_epoch_loss += batch_loss.item()
            _, predicted_classes = torch.max(model_output.data, 1)
            processed_samples += batch_labels.size(0)
            correct_predictions += (predicted_classes == batch_labels).sum().item()

        # Статистика эпохи
        actual_batches = min(len(data_loader),
                             training_config['max_batches'] or len(data_loader))
        average_loss = total_epoch_loss / actual_batches
        epoch_accuracy = 100 * correct_predictions / processed_samples

        print(f'Эпоха [{current_epoch + 1}/{training_config["epochs"]}], '
              f'Средние потери: {average_loss:.4f}, '
              f'Точность: {epoch_accuracy:.2f}%')

        if not quick_training:
            learning_rate_scheduler.step(average_loss)

    return classifier_model


def predict_avian_species(model_checkpoint_path: str,
                          test_images_directory: str) -> Dict[str, int]:
    """
    Классификация изображений птиц с использованием обученной модели

    Args:
        model_checkpoint_path: Путь к файлу модели
        test_images_directory: Директория с тестовыми изображениями

    Returns:
        Словарь с предсказанными классами для каждого изображения
    """

    computation_device = torch.device('cpu')

    # Загрузка чекпоинта модели
    model_checkpoint = torch.load(model_checkpoint_path,
                                  map_location=computation_device)

    # Определение архитектуры модели
    if isinstance(model_checkpoint, dict):
        species_count = model_checkpoint.get('species_count', None)
        model_parameters = model_checkpoint.get('model_parameters', model_checkpoint)
    else:
        model_parameters = model_checkpoint
        species_count = None

    # Автоматическое определение количества классов
    if species_count is None:
        species_count = model_parameters['species_predictor.weight'].shape[0]

    # Инициализация и загрузка модели
    prediction_model = AvianSpeciesClassifier(species_count,
                                              pretrained_weights=False)
    prediction_model.load_state_dict(model_parameters)
    prediction_model.to(computation_device)
    prediction_model.eval()

    # Трансформации для инференса
    inference_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Поиск изображений для классификации
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(test_images_directory)
                   if f.lower().endswith(supported_formats)]

    classification_results = {}

    # Обработка изображений
    with torch.no_grad():
        for image_filename in image_files:
            image_path = os.path.join(test_images_directory, image_filename)

            try:
                image = Image.open(image_path).convert('RGB')
                transformed_image = inference_transforms(image)
                batched_image = transformed_image.unsqueeze(0).to(computation_device)

                model_output = prediction_model(batched_image)
                _, predicted_species = torch.max(model_output, 1)
                classification_results[image_filename] = predicted_species.item()

            except Exception as processing_error:
                print(f"Ошибка обработки {image_filename}: {processing_error}")
                classification_results[image_filename] = 0

    return classification_results


def train_classifier(train_gt: Dict[str, int],
                     train_img_dir: str,
                     fast_train: bool = False):
    """Интерфейсная функция для обучения (совместимость с тестовой системой)"""
    return train_avian_classifier(train_gt, train_img_dir, fast_train)


def classify(model_path: str, test_img_dir: str) -> Dict[str, int]:
    """Интерфейсная функция для классификации (совместимость с тестовой системой)"""
    return predict_avian_species(model_path, test_img_dir)


if __name__ == '__main__':
    # Код для автономного тестирования
    import sys


    def load_training_labels(csv_file_path: str) -> Dict[str, int]:
        """Загрузка меток обучения из CSV файла"""
        labels_dict = {}
        try:
            with open(csv_file_path, 'r') as csv_file:
                next(csv_file)  # Пропуск заголовка
                for csv_line in csv_file:
                    file_name, class_label = csv_line.strip().split(',')
                    labels_dict[file_name] = int(class_label)
        except FileNotFoundError:
            print(f"Ошибка: файл {csv_file_path} не найден")
        return labels_dict


    # Автоматический поиск данных обучения
    possible_paths = [
        'tests/00_test_img_input/train/gt.csv',
        'tests/train/gt.csv',
        'train/gt.csv'
    ]

    found_training_labels = None
    found_image_directory = None

    for data_path in possible_paths:
        if os.path.exists(data_path):
            found_training_labels = load_training_labels(data_path)
            found_image_directory = os.path.join(os.path.dirname(data_path), 'images')
            print(f"Обнаружены данные: {data_path}")
            break

    if found_training_labels and found_image_directory:
        print("Запуск процесса обучения...")
        trained_model = train_avian_classifier(found_training_labels,
                                               found_image_directory,
                                               quick_training=False)

        # Сохранение модели
        species_count = max(found_training_labels.values()) + 1
        torch.save({
            'model_parameters': trained_model.state_dict(),
            'species_count': species_count
        }, 'birds_model.pt')

        print("Обучение завершено успешно! Модель сохранена.")
    else:
        print("Обучающие данные не обнаружены.")