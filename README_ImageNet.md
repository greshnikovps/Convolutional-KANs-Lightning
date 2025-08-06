# KAN Convolutional Models on ImageNet

Этот проект содержит реализацию конволюционных KAN (Kolmogorov-Arnold Networks) моделей для классификации изображений на датасете ImageNet, а также сравнение с обычными CNN моделями.

## Структура проекта

- `kan_convolutional/KANLightningModel.py` - Lightning модуль для KAN конволюционной модели
- `train_kan_imagenet.py` - Скрипт для обучения KAN модели на ImageNet
- `train_cnn_imagenet.py` - Скрипт для обучения обычной CNN модели на ImageNet
- `compare_models.py` - Скрипт для сравнения результатов обучения
- `requirements_imagenet.txt` - Зависимости для проекта

## Установка

1. Установите зависимости:
```bash
pip install -r requirements_imagenet.txt
```

2. Подготовьте датасет ImageNet:
   - Скачайте ImageNet датасет
   - Распакуйте в папку с следующей структурой:
```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Обучение KAN модели

```bash
python train_kan_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --grid_size 5 \
    --spline_order 3 \
    --gpus 1 \
    --precision 16
```

### Параметры KAN модели:
- `--grid_size`: Размер сетки для KAN (по умолчанию: 5)
- `--spline_order`: Порядок сплайна (по умолчанию: 3)
- `--scale_noise`: Масштаб шума (по умолчанию: 0.1)
- `--scale_base`: Масштаб базовой функции (по умолчанию: 1.0)
- `--scale_spline`: Масштаб сплайна (по умолчанию: 1.0)
- `--regularize_activation`: Регуляризация активации (по умолчанию: 1.0)
- `--regularize_entropy`: Регуляризация энтропии (по умолчанию: 1.0)

## Обучение CNN модели

```bash
python train_cnn_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --gpus 1 \
    --precision 16
```

## Сравнение моделей

После обучения обеих моделей, используйте скрипт сравнения:

```bash
python compare_models.py \
    --kan_checkpoint path/to/kan_model.ckpt \
    --cnn_checkpoint path/to/cnn_model.ckpt \
    --data_dir /path/to/imagenet \
    --create_plots
```

## Архитектура моделей

### KAN модель:
- 4 конволюционных слоя с KAN ядрами
- Размеры каналов: 3 → 64 → 128 → 256 → 512
- Размеры ядер: 7x7, 3x3, 3x3, 3x3
- Global Average Pooling
- KAN классификатор

### CNN модель:
- 4 конволюционных слоя с обычными ядрами
- Batch Normalization после каждого слоя
- Размеры каналов: 3 → 64 → 128 → 256 → 512
- Размеры ядер: 7x7, 3x3, 3x3, 3x3
- Global Average Pooling
- Линейный классификатор

## Мониторинг обучения

Используйте TensorBoard для мониторинга:

```bash
tensorboard --logdir logs
```

## Результаты

Скрипт сравнения выводит:
- Количество параметров каждой модели
- Размер модели в MB
- Loss, Top-1 и Top-5 точность
- Различия между моделями
- Сохраняет результаты в JSON файл
- Создает графики сравнения (опционально)

## Особенности KAN моделей

1. **Нелинейные ядра**: KAN ядра могут обучать сложные нелинейные функции
2. **Регуляризация**: Встроенная регуляризация через сплайны
3. **Интерпретируемость**: Возможность анализа обученных функций
4. **Адаптивность**: Автоматическая адаптация сетки к данным

## Требования к системе

- CUDA-совместимая GPU (рекомендуется)
- Минимум 16GB RAM
- Достаточно места на диске для ImageNet (~150GB)
- Python 3.8+

## Примечания

- Для полного обучения на ImageNet потребуется значительное время
- Рекомендуется использовать смешанную точность (precision=16)
- Модели автоматически сохраняются в папку `checkpoints/`
- Логи сохраняются в папку `logs/` для TensorBoard 