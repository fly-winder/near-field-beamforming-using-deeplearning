from collections import OrderedDict

import torch.nn as nn


class Net(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=8):
        super(Net, self).__init__()

        features = init_features
        self.encoder1 = Net._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder2 = Net._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder3 = Net._block(features * 2, features * 2, name="enc3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features * 2, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder2 = Net._block((features * 2), features * 1, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 1, features * 1, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder1 = Net._block(features * 1, out_channels, name="dec1")
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256 * 2, 256)
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        dec2 = self.upconv2(enc3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        dec1 = self.flatten(dec1)
        dec1 = self.linear(dec1)
        dec1 = self.tanh(dec1)
        return dec1
        # return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=2,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




flag = 0
if flag:
    model = Net()
    from torchsummary import summary

    summary(model, input_size=(1, 2, 256), device="cpu")
    print(model)