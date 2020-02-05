import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, height, width, channel, nbf, hidden_size):

        super(Generator, self).__init__()
        self.width = width
        self.height = height
        self.channel = channel
        self.nbf = nbf
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, self.nbf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nbf, self.nbf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nbf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nbf * 2, self.nbf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nbf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nbf * 4, self.nbf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nbf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.nbf * 8, self.nbf * 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(4608, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_size, self.height * self.width * self.channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img_a, img_b):

        hidden_a = self.encoder(img_a)
        hidden_b = self.encoder(img_b)
        hidden_sum = torch.cat([hidden_a, hidden_b], -1)
        mask = self.decoder(hidden_sum)
        mask = torch.reshape(mask, (self.channel, self.height, self.width))
        values, indices = torch.max(mask), torch.argmax(mask)
        masked_img = img_a.clone().flatten()
        masked_img[indices] = values
        masked_img = masked_img.reshape(img_a.shape)
        return masked_img
