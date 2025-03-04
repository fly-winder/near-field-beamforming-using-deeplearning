import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils_pytorch import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net

# Load data
path = 'train_set/data'
H, H_est = mat_load(path)

H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
H = np.squeeze(H)
H = torch.tensor(H, dtype=torch.complex32)
H_input = torch.tensor(H_input, dtype=torch.float32)

SNR = torch.pow(10, torch.randint(-20, 20, (H.shape[0], 1), dtype=torch.int32) / 10.0)

# Shuffle data
H_input, H = shuffle(H_input, H)

train_size = int(0.9 * len(H_input))
val_size = len(H_input) - train_size

# Create datasets and dataloaders
train_dataset = TensorDataset(H_input, H, SNR)
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
val_loader = DataLoader(val_data, batch_size=100, shuffle=True)



# Construct Model
input_size = H_input.shape[3]
model = Net()
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=20, min_lr=0.00005)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets, snr_values) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = Rate_func(targets,outputs,snr_values)
        loss = torch.mean(loss)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    learning_rate_scheduler.step(loss.item())
    avg_loss = running_loss / len(train_loader)
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets, s) in enumerate(val_loader):
            outputs = model(inputs)
            loss = Rate_func(targets, outputs, snr_values)
            loss = torch.mean(loss)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f}")



# Save the model
torch.save(model.state_dict(), 'mymodel.pth')

# Test the model
# rate = []
# model.eval()
# with torch.no_grad():
#     for snr in range(-20, 21, 5):
#         snr_values = torch.Tensor(10 ** (np.ones([H.shape[0], 1]) * snr / 10))
#         inputs = torch.Tensor(H_input)
#         targets = torch.Tensor(H)
#         outputs = model(inputs)
#
#         loss = Rate_func(targets,outputs,snr_values)
#         loss = torch.mean(loss)
#         print(f'SNR: {snr}, Loss: {loss.item():.4f}')
#         rate.append(-loss.item())
#
# a = outputs[1,:]
# numpy_array = a.numpy()
# np.savetxt('W.csv', numpy_array, delimiter=',')
# plt.figure()
# plt.xlabel("SNR(dB)")
# plt.ylabel("Spectral Efficiency")
# plt.plot(range(-20, 25, 5), rate)
# plt.show()
