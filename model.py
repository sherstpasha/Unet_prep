import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(
                features * 8 + features * 4, features * 4, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(
                features * 4 + features * 2, features * 2, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(features * 2 + features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        out = self.dec1(dec1)
        out = self.conv_final(out)
        out = self.sigmoid(out)
        return out


class DetectionHead(nn.Module):
    def __init__(self, in_channels=1, num_boxes=20):
        super(DetectionHead, self).__init__()
        self.num_boxes = num_boxes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_boxes * 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_boxes, 5)
        x = self.sigmoid(x)
        return x


class MultiStageModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32, num_boxes=20):
        super(MultiStageModel, self).__init__()
        self.unet = UNet(in_channels, out_channels, features)
        self.det_head = DetectionHead(in_channels=out_channels, num_boxes=num_boxes)

    def forward(self, x):
        preprocessed = self.unet(x)
        boxes = self.det_head(preprocessed)
        return preprocessed, boxes
