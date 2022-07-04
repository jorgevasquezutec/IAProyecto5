import torch.nn as nn


class CNNSimple(nn.Module):
    def __init__(self,num_clasess):
        super().__init__()
        # 3 140 140
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #64 140 140
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        #128 70 70
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        #256 35 35
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(7),
        )
        #256 7 7

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        #256 1 1
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256*5*5, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_clasess)
        )
  # el dropout es escoder de manera selectiva data al modelo con el fin
  # que el modelo , para que no acaba aprendiendo pixeles en especifico sino que aprenda
  # relaciones entre caracteristicas.

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.linear(out)
        return out