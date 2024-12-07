import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import logging
from pathlib import Path
from model import Net
from data import get_data_loaders
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def setup_logger():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (M1/M2)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    return device, device_name

def train(epochs=20, batch_size=64, learning_rate=0.01, target_accuracy=99.4):
    set_seed(42)  # Set seed for reproducibility
    setup_logger()
    
    # Device setup
    device, device_name = get_device()
    gpu_info = f"Using device: {device} ({device_name})"
    
    if device.type == 'cuda':
        gpu_info += f"\nMemory Usage:"
        gpu_info += f"\n  Allocated: {round(torch.cuda.memory_allocated(0)/1024**2,1)} MB"
        gpu_info += f"\n  Cached:    {round(torch.cuda.memory_reserved(0)/1024**2,1)} MB"
    
    print(gpu_info)
    logging.info(gpu_info)
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_loader, test_loader = get_data_loaders(batch_size)
    
    best_accuracy = 0.0
    early_stop = False
    
    for epoch in range(epochs):
        if early_stop:
            break
            
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print batch progress
            # if batch_idx % 100 == 0:
            #     print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
            #           f'({100. * batch_idx / len(train_loader):.0f}%)]  Loss: {loss.item():.6f}')
            
        train_accuracy = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        val_accuracy = 100. * correct / total
        val_loss = val_loss / len(test_loader)
        
        # Update best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
        # Check for early stopping
        if val_accuracy >= target_accuracy:
            early_stop = True
            print(f"\nReached target accuracy of {target_accuracy}% at epoch {epoch}")
            logging.info(f"Reached target accuracy of {target_accuracy}% at epoch {epoch}")
        
        # Log epoch results
        log_message = (f'Epoch: {epoch} | '
                      f'Train Loss: {train_loss:.3f} | '
                      f'Train Acc: {train_accuracy:.2f}% | '
                      f'Val Loss: {val_loss:.3f} | '
                      f'Val Acc: {val_accuracy:.2f}% | '
                      f'Best Val Acc: {best_accuracy:.2f}%')
        logging.info(log_message)
        print(log_message)
        
        # Log GPU memory only for CUDA devices
        if device.type == 'cuda':
            memory_info = (f"GPU Memory: "
                         f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**2,1)} MB, "
                         f"Cached: {round(torch.cuda.memory_reserved(0)/1024**2,1)} MB")
            logging.info(memory_info)
            print(memory_info)

    return best_accuracy

if __name__ == '__main__':
    final_accuracy = train()
    print(f"\nTraining completed. Best validation accuracy: {final_accuracy:.2f}%") 