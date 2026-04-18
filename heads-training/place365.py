'''Prompt
please write deep learning training code for places205 dataset.
The backbone is using swin transformer for feature extraction
the backbone and the head should be split into two parts
The head is fully connected layer for classification
try to be modular and reusable
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm

# Define transformations for the training and validation sets
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load the Places365 dataset
def get_dataloaders(batch_size=512, num_workers=8):
    transform = get_transforms()
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'data_large_standard')
    val_dir = os.path.join(data_dir, 'val_large')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        train_dataset = datasets.Places365(root=data_dir, split='train-standard', download=True, transform=transform)
        val_dataset = datasets.Places365(root=data_dir, split='val', download=True, transform=transform)
    else:
        train_dataset = datasets.Places365(root=data_dir, split='train-standard', download=False, transform=transform)
        val_dataset = datasets.Places365(root=data_dir, split='val', download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# Define the model with Swin Transformer backbone and fully connected head
class Places365Model(nn.Module):
    def __init__(self, num_classes=365):
        super(Places365Model, self).__init__()
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        # self.head = nn.Linear(self.backbone.num_features, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=lr)  # Only optimize the head parameters

    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
        
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(val_loader, desc='Validation', unit='batch')
        
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save the head parameters separately
            torch.save(model.head.state_dict(), 'best_head_place365.pth')
            torch.save(model.state_dict(), 'best_model_place365.pth')
            print('Best model saved with accuracy: {:.2f}%'.format(best_accuracy))
    
    print('Training complete')

# Main function to run the training
def main():
    train_loader, val_loader = get_dataloaders()
    model = Places365Model()
    train_model(model, train_loader, val_loader, gpu_id=1)  # Specify the GPU ID here

if __name__ == '__main__':
    main()