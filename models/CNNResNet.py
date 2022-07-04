import torch.nn as nn

class CNNResNet(nn.Module):
  def __init__(self,num_clasess):
        super().__init__()
        # 3 140 140
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 64 140 140
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 128 70 70
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 128 70 70
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 256 35 35 
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(7)
        )
        # 512 5 5
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.MaxPool2d(5),
            nn.Flatten(),
            # nn.Dropout(0.2),
            nn.Linear(512, num_clasess)
        )

  def forward(self,x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.res1(out) + out
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.res2(out) + out
      out = self.linear(out)
      return out
