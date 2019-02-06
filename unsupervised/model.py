import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        conv_dim = 64
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5, 2, 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 5, 2, 2)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 1, 0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        # Encoder
        x = F.leaky_relu(self.conv1(x), 0.05)
        x = F.leaky_relu(self.conv2(x), 0.05)

        x = F.leaky_relu(self.conv3(x), 0.05)
        x = F.leaky_relu(self.conv4(x), 0.05)
        x = x.view(-1, 512)

        # Classifier
        x_c = F.relu(self.fc1(x))
        x_c = F.dropout(x_c, training=self.training)
        x_c = self.fc2(x_c)
        x_c = F.dropout(x_c, training=self.training)
        x_c = self.fc3(x_c)
        x_c = F.dropout(x_c, training=self.training)
        x_c = self.fc4(x_c)

        return x, F.softmax(x_c, dim=1)


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        # Discriminator
        x_d = F.relu(self.fc1(x))
        x_d = F.dropout(x_d, training=self.training)
        x_d = self.fc2(x_d)
        x_d = F.dropout(x_d, training=self.training)
        x_d = self.fc3(x_d)
        x_d = F.dropout(x_d, training=self.training)
        x_d = self.fc4(x_d)

        return torch.sigmoid(x_d)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True))  # bias=False
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)
