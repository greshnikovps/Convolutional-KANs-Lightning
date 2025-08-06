# Training KAN and CNN Models on ImageNet

Этот проект содержит скрипты для обучения KAN (Kolmogorov-Arnold Networks) и CNN моделей на ImageNet.

## Модели

### KAN Model
- **Параметры**: ~378K (1.44 MB)
- **Архитектура**: 2 конволюционных слоя (16->32 каналов)
- **Особенности**: Использует KAN-специфичные слои с B-spline активациями

### CNN Model  
- **Параметры**: ~39K (0.15 MB)
- **Архитектура**: 2 конволюционных слоя (16->32 каналов)
- **Особенности**: Стандартные CNN слои с BatchNorm

## Быстрое обучение (для тестирования)

### Обучение KAN модели
```bash
python train_kan_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500
```

### Обучение CNN модели
```bash
python train_cnn_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500
```

## Полное обучение

### Обучение KAN модели
```bash
python train_kan_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32
```

### Обучение CNN модели
```bash
python train_cnn_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32
```

## Параметры

### Общие параметры
- `--data_dir`: Путь к датасету ImageNet
- `--batch_size`: Размер батча (по умолчанию 16 для small, 32 для полного)
- `--epochs`: Количество эпох
- `--learning_rate`: Скорость обучения (по умолчанию 1e-3)
- `--gpus`: Количество GPU (по умолчанию 1)
- `--precision`: Точность обучения 16 или 32 (по умолчанию 32)

### KAN-специфичные параметры
- `--grid_size`: Размер сетки для KAN слоев (по умолчанию 5)
- `--spline_order`: Порядок B-spline (по умолчанию 3)
- `--regularize_activation`: Вес регуляризации активации (по умолчанию 1.0)
- `--regularize_entropy`: Вес регуляризации энтропии (по умолчанию 1.0)

### Параметры архитектуры
- `--layers`: Конфигурация слоев в формате "out_channels,kernel_size,stride,padding;..."
- `--subset_size`: Размер подмножества для быстрого обучения (только для small скриптов)

## Примеры использования

### Быстрое тестирование на небольшом подмножестве
```bash
# KAN модель на 500 образцах, 5 эпох
python train_kan_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500 --gpus 0

# CNN модель на 500 образцах, 5 эпох  
python train_cnn_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500 --gpus 0
```

### Обучение с кастомной архитектурой
```bash
# KAN модель с кастомными слоями
python train_kan_small.py --data_dir /path/to/imagenet --layers "8,3,1,1;16,3,2,1" --epochs 10

# CNN модель с кастомными слоями
python train_cnn_small.py --data_dir /path/to/imagenet --layers "8,3,1,1;16,3,2,1" --epochs 10
```

### Полное обучение на GPU
```bash
# KAN модель на полном датасете
python train_kan_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32 --gpus 1

# CNN модель на полном датасете
python train_cnn_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32 --gpus 1
```

## Мониторинг

Все эксперименты логируются в TensorBoard:
```bash
tensorboard --logdir logs
```

## Результаты

Модели сохраняются в:
- `checkpoints/kan_small/` - чекпоинты KAN модели
- `checkpoints/cnn_small/` - чекпоинты CNN модели
- `logs/` - логи TensorBoard

## Сравнение моделей

После обучения используйте скрипт `compare_models.py` для сравнения результатов:
```bash
python compare_models.py --kan_checkpoint path/to/kan.ckpt --cnn_checkpoint path/to/cnn.ckpt --data_dir /path/to/imagenet
```

## Требования

- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- TorchMetrics >= 1.0.0
- ImageNet датасет

## Структура ImageNet

Убедитесь, что ваш ImageNet датасет имеет следующую структуру:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
``` 