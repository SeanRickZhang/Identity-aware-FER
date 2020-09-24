import torch
import torch.nn as nn


class IAFER_model(nn.Module):
    def __init__(self, channel_num):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_num, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier1 = nn.Sequential(
            nn.Linear(512, 1024),
            #nn.Dropout(p = 0.5),
            nn.Linear(1024, 1024),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, Xs, xreal):
        outputs = []
        output_hinges = []
        for x in Xs:
            output = self.conv1(x)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.conv4(output)
            output = self.avg_pool(output)
            output = output.view(-1, 512)
            outputs.append(output)

        outputr = self.conv1(xreal)
        outputr = self.conv2(outputr)
        outputr = self.conv3(outputr)
        outputr = self.conv4(outputr)
        outputr = self.avg_pool(outputr)
        outputr = outputr.view(-1, 512)

        for output_feature in outputs:
            output_hinge = self.classifier1(output_feature - outputr)
            output_hinges.append(output_hinge)

        return output_hinges, outputs, outputr

def create_IAFER_model():
    return IAFER_model(channel_num = 1)