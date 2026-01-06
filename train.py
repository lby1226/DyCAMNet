import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from DyCAMNet import DyCAMNet, NetworkConfig
import random
from tqdm import tqdm
from config import Config

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Dataset
class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load normal category (label 0)
        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)
        
        # Load unnormal category (label 1)
        unnormal_dir = os.path.join(data_dir, 'unnormal')
        if os.path.exists(unnormal_dir):
            for img_name in os.listdir(unnormal_dir):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(unnormal_dir, img_name))
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': total_loss/len(train_loader), 'Acc': 100.*correct/total})
    
    return total_loss/len(train_loader), 100.*correct/total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': total_loss/len(val_loader), 'Acc': 100.*correct/total})
    
    return total_loss/len(val_loader), 100.*correct/total

def main():
    # Set random seed
    set_seed(Config.train_config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data preprocessing
    transform_config = Config.get_train_transform_config()
    transform = transforms.Compose([
        transforms.Resize((transform_config['img_size'], transform_config['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(transform_config['random_rotate']),
        transforms.ColorJitter(**transform_config['color_jitter']),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_config['normalize_mean'], 
                           std=transform_config['normalize_std'])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((transform_config['img_size'], transform_config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_config['normalize_mean'], 
                           std=transform_config['normalize_std'])
    ])
    
    # Load dataset
    trainval_dataset = Dataset(Config.data_config['train_dir'], transform=transform)
    
    # Split training and validation sets
    train_size = int(Config.data_config['train_val_split'] * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.train_config['batch_size'],
        shuffle=True,
        num_workers=Config.data_config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.train_config['batch_size'],
        shuffle=False,
        num_workers=Config.data_config['num_workers']
    )
    
    # Create model
    model_config = NetworkConfig(
        layers=Config.model_config['block_layers'],
        num_classes=Config.model_config['num_classes'],
        zero_init_residual=Config.model_config['zero_init_residual'],
        groups=Config.model_config['groups'],
        width_per_group=Config.model_config['width_per_group']
    )
    model = DyCAMNet(model_config)
    
    # Pretrained weights loading removed as per request
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.train_config['base_lr'],
        weight_decay=Config.train_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.train_config['T_max'],
        eta_min=Config.train_config['min_lr']
    )
    
    # Training parameters
    num_epochs = Config.train_config['num_epochs']
    best_val_acc = 0
    
    # Create save directory
    os.makedirs(Config.save_config['save_dir'], exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(Config.save_config['save_dir'], Config.save_config['model_name'])
            # Save model parameters only
            torch.save(model.state_dict(), save_path)
            
            # Save training state separately (optional)
            train_state_path = os.path.join(Config.save_config['save_dir'], 'train_state.pth')
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, train_state_path)
        
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%')
        print(f'Current LR: {scheduler.get_last_lr()[0]:.6f}')

if __name__ == '__main__':
    main() 