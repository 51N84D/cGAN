from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F

class cGAN(nn.Module):
    def __init__(self):
        super(cGAN, self).__init__()
        #encoder
        self.conv1 = nn.Conv2d(1, 32, (4, 4), stride=(2,3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (4, 9), stride=(2,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 2, (6, 6), stride=4, padding=1)
        self.bn3 = nn.BatchNorm2d(2)
        #decoder
        self.conv4 = nn.ConvTranspose2d(16, 32, (4, 4), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 3, (4, 4), stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.2)   

    def encode(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p =0.5)
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=0.5)
        x = x.view(-1, 128)
        return x	


    def decode(self, z):
        z = z.view(-1, 16, 4, 4)
        z = self.lrelu(self.bn4(self.conv4(z)))
        z = F.dropout(z, p=0.5)
        z = self.lrelu(self.bn5(self.conv5(z)))
        z = torch.tanh(self.conv6(z))
        return z

    def forward(self, noise, spec):
        #Encode spectrogram
        z = self.encode(spec)
        #Concat noise
        x = torch.cat([z, noise], -1)
        #Decode
        x = self.decode(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(DiscriminatorDropout, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=6, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(6)
        #----------------------------
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.lrelu = nn.LeakyReLU(negative_slope = 0.2)

    def forward(self, img, labels):
        #First process image into latent vector
        x = self.lrelu(self.conv1(img))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = x.view(-1, 5*5*6)
        c = self.label_emb(labels)
        x = torch.cat([x, c], -1)
        x = x.view(-1, 10, 4, 4)
        #--------------Decode
        x = self.lrelu(self.bn3(self.conv3(x)))
        #x = F.dropout(x, p=0.5)
        x = self.lrelu(self.bn4(self.conv4(x)))
        #x = F.dropout(x, p=0.5)
        x = self.lrelu(self.bn5(self.conv5(x)))
        #x = F.dropout(x, p=0.5)
        x = self.lrelu(self.conv6(x))
        x = x.view(-1,1)
        #x = torch.sigmoid(x)

        return x


