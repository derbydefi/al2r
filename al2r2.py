import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


torch.manual_seed(random.randint(1,1e3))  # Set a random seed for reproducibility

#load mnist dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#define simple nn
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_control = SimpleNN().to(device)
model_experimental = SimpleNN().to(device)
#model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()

# Initial learning rate and regularization strength
learning_rate = 0.01
reg_strength = 0.01

optimizer_control = optim.SGD(model_control.parameters(), lr=learning_rate, weight_decay=reg_strength)
optimizer_experimental = optim.SGD(model_experimental.parameters(), lr=learning_rate, weight_decay=reg_strength)

def train_model(model, optimizer, epochs, adaptive_reg=False, reg_adjust_threshold=0.01):
    # Initialize lists to keep track of losses and accuracies
    train_losses, val_losses, reg_strengths, train_accuracy, val_accuracy = [], [], [], [], []

    prev_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate training accuracy within the same loop
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc = 100 * correct / total
        train_accuracy.append(train_acc)

        val_loss = validate_model(model)
        val_losses.append(val_loss)
        val_acc = calculate_accuracy(model, test_loader)
        val_accuracy.append(val_acc)

        if adaptive_reg:
            if val_loss > prev_val_loss + reg_adjust_threshold:
                optimizer.param_groups[0]['weight_decay'] *= 1.1  # Increase regularization
            elif val_loss < prev_val_loss:
                optimizer.param_groups[0]['weight_decay'] *= 0.9  # Decrease regularization

        reg_strengths.append(optimizer.param_groups[0]['weight_decay'])
        prev_val_loss = val_loss

        print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%, Regularization: {optimizer.param_groups[0]["weight_decay"]:.4f}')
    
    return train_losses, val_losses, reg_strengths, train_accuracy, val_accuracy
def validate_model(model):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
    return val_loss / len(test_loader)
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total
def model_complexity(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_norm = sum(p.norm().item() for p in model.parameters())
    return total_params, total_norm

total_params, total_norm = model_complexity(model_control)
print(f"Control Model has {total_params} parameters and a norm of {total_norm}.")
total_params, total_norm = model_complexity(model_experimental)
print(f"Experimental Model has {total_params} parameters and a norm of {total_norm}.")


# Train the Control Model
train_losses_control, val_losses_control, reg_strengths_control, train_accuracy_control, val_accuracy_control = train_model(model_control, optimizer_control, epochs=10, adaptive_reg=False)

# Train the Experimental Model
train_losses_exp, val_losses_exp, reg_strengths_exp, train_accuracy_exp, val_accuracy_exp = train_model(model_experimental, optimizer_experimental, epochs=10, adaptive_reg=True)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses_control, label='Control Training Loss')
plt.plot(train_losses_exp, label='Experimental Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(val_losses_control, label='Control Validation Loss')
plt.plot(val_losses_exp, label='Experimental Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_accuracy_control, label='Control Train Accuracy')
plt.plot(train_accuracy_exp, label='Experimental Train Accuracy')
plt.title('Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(val_accuracy_control, label='Control Validation Accuracy')
plt.plot(val_accuracy_exp, label='Experimental Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()

plt.tight_layout()
plt.show()

