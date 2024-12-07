import torch.nn as nn
import torch.nn.functional as F

DROP_OUT = 0.05

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROP_OUT)
        )  # output_size = 26, receptive_field = 3, output_channels = 16

        # Convolution Block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(DROP_OUT)
        )  # output_size = 24, receptive_field = 5, output_channels = 32

        # Transition Block 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 12, 1, padding=0, bias=False),
        )  # output_size = 24, receptive_field = 5, output_channels = 12
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12, receptive_field = 6, output_channels = 14

        # Convolution Block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROP_OUT)
        )  # output_size = 10, receptive_field = 10, output_channels = 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROP_OUT)
        )  # output_size = 8, receptive_field = 14, output_channels = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROP_OUT)
        )  # output_size = 6, receptive_field = 18, output_channels = 16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROP_OUT)
        )  # output_size = 6, receptive_field = 22, output_channels = 16

        # Fully Connected Layer
        self.fc = nn.Linear(16 * 6 * 6, 10)  # 16 channels, 6x6 feature map

        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(-1, 16 * 6 * 6)  # Flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)