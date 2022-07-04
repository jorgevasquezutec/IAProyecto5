import torch.nn as nn

import torch.nn as nn

class CNNSmall(nn.Module):
    def __init__(self,num_clasess):
        super().__init__()
        # 3 140 140
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #32x140x140
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        #64,70,70
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        #128,35,35
        #128,7,7
        self.linear = nn.Sequential(
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128*7*7, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_clasess)
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        out = self.linear(out)
        return out