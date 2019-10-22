from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        #encoder
        self.conv1 = nn.Conv2d(1, 32, (4, 4), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (4, 4), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, (4, 4), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, (4, 4), stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.fc5 = nn.Linear(256*8*18, 256)
        self.fc61 = nn.Linear(256, 128)
        self.fc62 = nn.Linear(256, 128)
        #decoder
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 64*4*4)
        self.conv8 = nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv10 = nn.ConvTranspose2d(32, 32, (4, 4), stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(32)
        self.conv11 = nn.ConvTranspose2d(32, 32, (3, 3), stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.conv12 = nn.ConvTranspose2d(32, 32, (2, 2), stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.conv13 = nn.ConvTranspose2d(32, 3, (4, 4), stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, 256 * 8 * 18)
        x = F.relu(self.fc5(x))
        return self.fc61(x), self.fc62(x)	

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = z.view(-1, 64, 4, 4)
        z = F.relu(self.bn8(self.conv8(z)))
        z = F.relu(self.bn9(self.conv9(z)))
        z = F.relu(self.bn10(self.conv10(z)))
        z = F.relu(self.bn11(self.conv11(z)))
        z = F.relu(self.bn12(self.conv12(z)))
        z = torch.sigmoid(self.conv13(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

