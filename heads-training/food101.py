import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# Define transformations for the training and validation sets
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load the Food101 dataset
def get_dataloaders(batch_size=512, num_workers=4):
    transform = get_transforms()
    train_dataset = datasets.Food101(root='data', split='train', download=True, transform=transform)
    val_dataset = datasets.Food101(root='data', split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# Define the model with Swin Transformer backbone and fully connected head
class Food101Model(nn.Module):
    def __init__(self, num_classes=101):
        super(Food101Model, self).__init__()
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
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
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            torch.save(model.head.state_dict(), 'best_head_food101.pth')
            torch.save(model.state_dict(), 'best_model_food101.pth')
            print('Best model saved with accuracy: {:.2f}%'.format(best_accuracy))
    
    print('Training complete')

# Main function to run the training
def main():
    train_loader, val_loader = get_dataloaders()
    model = Food101Model()
    train_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main()    main()