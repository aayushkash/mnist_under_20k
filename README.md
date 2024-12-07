# MNIST CNN Classifier

[![Python Tests](https://github.com/aayushkash/mnist_under_20k/actions/workflows/python-app.yml/badge.svg)](https://github.com/aayushkash/mnist_under_20k/actions/workflows/python-app.yml)

This project implements a Convolutional Neural Network (CNN) for classifying MNIST digits using PyTorch. The model is designed to be lightweight with less than 20,000 parameters while maintaining good accuracy.

## Features

- Lightweight CNN architecture (<20K parameters)
- Batch Normalization for better training stability
- Dropout for regularization
- Global Average Pooling to reduce parameters
- Comprehensive logging system
- Unit tests compatible with GitHub Actions

## Project Structure
mnist_cnn/
├── src/
│ ├── model.py # CNN architecture
│ ├── train.py # Training logic
│ ├── data.py # Data loading utilities
│ └── utils.py # Helper functions
├── tests/ # Unit tests
├── logs/ # Training logs
└── requirements.txt

### Key Results
- Reaches 99.4% validation accuracy in just 13 epochs
- Final validation accuracy: 99.46%
- Consistent improvement in both training and validation metrics
- Early stopping implemented at target accuracy of 99.4%

## Installation
pip install -r requirements.txt

## Training Results

The model achieves excellent performance on M1/M2 Mac using MPS acceleration:

## Usage

To train the model:

```bash
python train.py
```
To run tests:

```bash
python test.py
```

## Model Architecture
- 3 convolutional blocks with batch normalization
- Dropout regularization (p=0.05)
- SGD optimizer with momentum
- NLL Loss function
- Data augmentation with random rotation


## License

MIT License

## Training Logs
2024-12-07 20:30:32,332 - Using device: mps (Apple Silicon (M1/M2))
2024-12-07 20:30:50,286 - Epoch: 0 | Train Loss: 0.126 | Train Acc: 96.09% | Val Loss: 2.926 | Val Acc: 98.54% | Best Val Acc: 98.54%
2024-12-07 20:31:06,569 - Epoch: 1 | Train Loss: 0.050 | Train Acc: 98.42% | Val Loss: 2.354 | Val Acc: 98.83% | Best Val Acc: 98.83%
2024-12-07 20:31:23,068 - Epoch: 2 | Train Loss: 0.040 | Train Acc: 98.77% | Val Loss: 1.685 | Val Acc: 99.14% | Best Val Acc: 99.14%
2024-12-07 20:31:39,048 - Epoch: 3 | Train Loss: 0.035 | Train Acc: 98.93% | Val Loss: 1.666 | Val Acc: 99.16% | Best Val Acc: 99.16%
2024-12-07 20:31:54,991 - Epoch: 4 | Train Loss: 0.031 | Train Acc: 99.05% | Val Loss: 1.777 | Val Acc: 99.07% | Best Val Acc: 99.16%
2024-12-07 20:32:11,452 - Epoch: 5 | Train Loss: 0.027 | Train Acc: 99.12% | Val Loss: 1.558 | Val Acc: 99.27% | Best Val Acc: 99.27%
2024-12-07 20:32:27,807 - Epoch: 6 | Train Loss: 0.024 | Train Acc: 99.20% | Val Loss: 1.683 | Val Acc: 99.22% | Best Val Acc: 99.27%
2024-12-07 20:32:43,977 - Epoch: 7 | Train Loss: 0.021 | Train Acc: 99.32% | Val Loss: 1.553 | Val Acc: 99.18% | Best Val Acc: 99.27%
2024-12-07 20:33:00,369 - Epoch: 8 | Train Loss: 0.022 | Train Acc: 99.27% | Val Loss: 1.508 | Val Acc: 99.31% | Best Val Acc: 99.31%
2024-12-07 20:33:16,816 - Epoch: 9 | Train Loss: 0.019 | Train Acc: 99.39% | Val Loss: 1.579 | Val Acc: 99.23% | Best Val Acc: 99.31%
2024-12-07 20:33:33,094 - Epoch: 10 | Train Loss: 0.020 | Train Acc: 99.38% | Val Loss: 1.300 | Val Acc: 99.33% | Best Val Acc: 99.33%
2024-12-07 20:33:49,730 - Epoch: 11 | Train Loss: 0.017 | Train Acc: 99.46% | Val Loss: 1.514 | Val Acc: 99.23% | Best Val Acc: 99.33%
2024-12-07 20:34:06,813 - Epoch: 12 | Train Loss: 0.018 | Train Acc: 99.38% | Val Loss: 1.285 | Val Acc: 99.35% | Best Val Acc: 99.35%
2024-12-07 20:34:23,260 - Reached target accuracy of 99.4% at epoch 13
2024-12-07 20:34:23,260 - Epoch: 13 | Train Loss: 0.016 | Train Acc: 99.45% | Val Loss: 1.210 | Val Acc: 99.46% | Best Val Acc: 99.46%
