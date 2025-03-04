import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils_pytorch import *
from model import Net



# load model
model = Net()
# model = BFNN(input_size=(1,2,Nt), hidden_size1=256, hidden_size2=128, output_size=Nt)
model.load_state_dict(torch.load('mymodel.pth'))
model.eval()


path = 'train_set/data'
H, H_est = mat_load(path)
H = H_est
H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
H = np.squeeze(H)
H = torch.tensor(H, dtype=torch.complex32)
H_input = torch.tensor(H_input, dtype=torch.float32)
H_input, H = shuffle(H_input, H)

# test
rate = []
with torch.no_grad():
    for snr in range(-20, 21, 5):
        snr_values = torch.tensor(10 ** (np.ones([H.shape[0], 1]) * snr / 10), dtype=torch.float32)
        inputs = H_input.clone()
        targets = H.clone()

        outputs = model(inputs)
        loss = Rate_func(targets, outputs, snr_values)
        loss = torch.mean(loss)

        print(f'SNR: {snr} dB, Loss: {loss.item():.4f}')
        rate.append(-loss.item())

print("Rate:", rate)


plt.figure()
plt.plot(range(-20, 21, 5), rate, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Achievable Rate")
plt.title("Performance Evaluation")
plt.grid()
plt.show()
