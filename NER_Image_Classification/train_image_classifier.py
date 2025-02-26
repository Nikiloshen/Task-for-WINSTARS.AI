import argparse
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os  # Add this import

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='./image_model')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)  # Added this line

    # Rest of the code remains the same
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f'{args.data_dir}/train', train_transform)
    val_dataset = datasets.ImageFolder(f'{args.data_dir}/val', val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Update model loading to avoid deprecation warnings
    model = models.resnet18(weights='IMAGENET1K_V1')  # Changed from pretrained=True
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Rest of training loop remains unchanged
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    best_acc = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        val_acc = correct.double() / len(val_dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')

if __name__ == '__main__':
    main()