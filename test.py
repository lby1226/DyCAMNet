import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from train import Dataset
from DyCAMNet import DyCAMNet, NetworkConfig
from config import Config
import time
from thop import profile
from ptflops import get_model_complexity_info

def check_dataset(data_dir):
    """Check if the dataset exists and is not empty"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    normal_dir = os.path.join(data_dir, 'normal')
    unnormal_dir = os.path.join(data_dir, 'unnormal')
    
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Normal category directory does not exist: {normal_dir}")
    if not os.path.exists(unnormal_dir):
        raise FileNotFoundError(f"Unnormal category directory does not exist: {unnormal_dir}")
    
    normal_images = [f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    unnormal_images = [f for f in os.listdir(unnormal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not normal_images:
        raise ValueError(f"Normal category directory is empty: {normal_dir}")
    if not unnormal_images:
        raise ValueError(f"Unnormal category directory is empty: {unnormal_dir}")
    
    print(f"Dataset statistics:")
    print(f"- Normal category: {len(normal_images)} images")
    print(f"- Unnormal category: {len(unnormal_images)} images")
    print(f"- Total: {len(normal_images) + len(unnormal_images)} images")

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def visualize_predictions(model, test_loader, device, num_samples=10, save_dir='pred_samples'):
    """Visualize prediction results"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created prediction results save directory: {save_dir}")
    
    model.eval()
    # Denormalization transform
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(Config.train_config['normalize_mean'], 
                                 Config.train_config['normalize_std'])],
        std=[1/s for s in Config.train_config['normalize_std']]
    )
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Convert back to image format and save
            for j, (img, pred, true, prob) in enumerate(zip(images, predicted, labels, probs)):
                img = denorm(img).cpu()
                img = transforms.ToPILImage()(img)
                
                # Get prediction probability
                pred_prob = prob[pred].item()
                
                # Save image
                status = 'correct' if pred == true else 'wrong'
                filename = f'sample_{i}_{j}_pred_{pred.item()}({pred_prob:.2f})_true_{true.item()}_{status}.png'
                save_path = os.path.join(save_dir, filename)
                img.save(save_path)
                print(f"Saved prediction sample: {filename}")

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_size, device, num_iterations=100):
    """Measure inference time"""
    model.eval()
    x = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure time
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return elapsed_time / num_iterations

def print_size_of_model(model):
    """Print model size"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

def print_model_statistics(model, input_size, device):
    """Print model statistics"""
    # Calculate parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs
    input = torch.randn(1, 3, input_size, input_size).to(device)
    macs, _ = profile(model, inputs=(input,))
    flops = macs * 2  # FLOPs ≈ 2 * MACs
    
    # Measure inference time
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input)
    
    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    avg_time = np.mean(times) * 1000  # Convert to ms
    
    # Print table
    print("\n" + "="*50)
    print(f"{'Para(M)':^15} | {'FLOPs(G)':^15} | {'Time spent(ms)':^15}")
    print("-"*50)
    print(f"{num_params/1e6:^15.2f} | {flops/1e9:^15.2f} | {avg_time:^15.2f}")
    print("="*50)
    
    return {
        'params': num_params,
        'flops': flops,
        'time': avg_time
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check dataset
    test_dir = Config.data_config['test_dir']
    print(f"\nChecking test dataset: {test_dir}")
    check_dataset(test_dir)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((Config.data_config['img_size'], 
                         Config.data_config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.train_config['normalize_mean'],
                           std=Config.train_config['normalize_std'])
    ])
    
    # Load test dataset
    test_dataset = Dataset(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=Config.data_config['num_workers']
    )
    
    # Check model file
    model_path = os.path.join(Config.save_config['save_dir'], 
                             Config.save_config['model_name'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    
    model_config = NetworkConfig(
        layers=Config.model_config['block_layers'],
        num_classes=Config.model_config['num_classes'],
        zero_init_residual=Config.model_config.get('zero_init_residual', False),
        groups=Config.model_config.get('groups', 1),
        width_per_group=Config.model_config.get('width_per_group', 64)
    )
    model = DyCAMNet(model_config)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Print model statistics
    input_size = Config.data_config['img_size']
    stats = print_model_statistics(model, input_size, device)
    
    # Create results save directory
    results_dir = os.path.join(Config.test_config['pred_save_dir'], 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save performance metrics
    performance_path = os.path.join(results_dir, 'model_performance.txt')
    with open(performance_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"{'Para(M)':^15} | {'FLOPs(G)':^15} | {'Time spent(ms)':^15}\n")
        f.write("-"*50 + "\n")
        f.write(f"{stats['params']/1e6:^15.2f} | {stats['flops']/1e9:^15.2f} | {stats['time']:^15.2f}\n")
        f.write("="*50 + "\n")
    
    # Evaluation metrics
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0
    correct = 0
    total = 0
    
    # Test loop
    print('\nStarting test...')
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing progress'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    if len(test_loader) == 0:
        raise ValueError("Test dataset is empty!")
        
    test_loss = total_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    # Print results
    print(f'\nTest Results:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.2f}%')
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm, 
        classes=['Normal', 'Unnormal'],
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # Display detailed classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=['Normal', 'Unnormal'],
        digits=4
    )
    print('\nClassification Report:')
    print(report)
    
    # Save classification report
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('Test Results:\n')
        f.write(f'Loss: {test_loss:.4f}\n')
        f.write(f'Accuracy: {test_acc:.2f}%\n\n')
        f.write('Classification Report:\n')
        f.write(report)
    print(f"\nClassification report saved to: {report_path}")
    
    # Visualize some prediction results
    print('\nSaving prediction sample visualizations...')
    visualize_predictions(
        model, 
        test_loader, 
        device,
        save_dir=os.path.join(results_dir, 'pred_samples')
    )
    
    print(f'\nTest completed! All results saved to: {results_dir}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")