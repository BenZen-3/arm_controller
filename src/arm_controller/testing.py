# # import torch
# # import torch.nn as nn
# # import torchvision.transforms.functional as TF

# # # SOURCE: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

# # class DoubleConv(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super(DoubleConv, self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),
# #         )

# #     def forward(self, x):
# #         return self.conv(x)

# # class UNET(nn.Module):
# #     def __init__(
# #             self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
# #     ):
# #         super(UNET, self).__init__()
# #         self.ups = nn.ModuleList()
# #         self.downs = nn.ModuleList()
# #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# #         # Down part of UNET
# #         for feature in features:
# #             self.downs.append(DoubleConv(in_channels, feature))
# #             in_channels = feature

# #         # Up part of UNET
# #         for feature in reversed(features):
# #             self.ups.append(
# #                 nn.ConvTranspose2d(
# #                     feature*2, feature, kernel_size=2, stride=2,
# #                 )
# #             )
# #             self.ups.append(DoubleConv(feature*2, feature))

# #         self.bottleneck = DoubleConv(features[-1], features[-1]*2)
# #         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

# #     def forward(self, x):
# #         skip_connections = []

# #         for down in self.downs:
# #             x = down(x)
# #             skip_connections.append(x)
# #             x = self.pool(x)

# #         x = self.bottleneck(x)
# #         skip_connections = skip_connections[::-1]

# #         for idx in range(0, len(self.ups), 2):
# #             x = self.ups[idx](x)
# #             skip_connection = skip_connections[idx//2]

# #             if x.shape != skip_connection.shape:
# #                 x = TF.resize(x, size=skip_connection.shape[2:])

# #             concat_skip = torch.cat((skip_connection, x), dim=1)
# #             x = self.ups[idx+1](concat_skip)

# #         return self.final_conv(x)

# # def test():
# #     x = torch.randn((3, 1, 161, 161))
# #     model = UNET(in_channels=1, out_channels=1)
# #     preds = model(x)
# #     assert preds.shape == x.shape

# # if __name__ == "__main__":
# #     test()
# #     print('test complete')


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv3D, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UNet3D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, num_input_frames=10, features=[32, 64, 128, 256]):
#         super(UNet3D, self).__init__()
#         self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Pooling only spatially

#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()

#         # Down-sampling (Encoder)
#         for feature in features:
#             self.downs.append(DoubleConv3D(in_channels, feature))
#             in_channels = feature

#         # Bottleneck
#         self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

#         # Up-sampling (Decoder)
#         for feature in reversed(features):
#             self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
#             self.ups.append(DoubleConv3D(feature * 2, feature))

#         self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         skip_connections = []

#         # Encoder (Downsampling)
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]  # Reverse for decoding

#         # Decoder (Upsampling)
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx // 2]

#             # Ensure tensor sizes match for concatenation
#             if x.shape != skip_connection.shape:
#                 x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)

#             x = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx + 1](x)

#         return self.final_conv(x).squeeze(2)  # Remove temporal dimension if needed

# def test():
#     num_input_frames = 10
#     x = torch.randn((3, 1, num_input_frames, 64, 64))  # (batch, channels, time, height, width)
#     model = UNet3D(in_channels=1, out_channels=1, num_input_frames=num_input_frames)
#     preds = model(x)
    
#     assert preds.shape == (3, 1, num_input_frames, 64, 64), f"Expected shape {(3, 1, num_input_frames, 64, 64)}, but got {preds.shape}"

# if __name__ == "__main__":
#     test()
#     print("Test complete")


import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_input_frames=10, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # Pool across all dimensions

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down-sampling (Encoder)
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Up-sampling (Decoder)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=(2, 2, 2), stride=(2, 2, 2)))
            self.ups.append(DoubleConv3D(feature * 2, feature))

        # Additional Conv3D layers to squeeze temporal dimension to 1
        self.final_conv_layers = nn.Sequential(
            nn.Conv3d(features[0], features[0] // 2, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features[0] // 2, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(num_input_frames, 1, 1), stride=(num_input_frames, 1, 1))  # Final squeeze
        )

    def forward(self, x):
        skip_connections = []

        # Encoder (Downsampling)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoding

        # Decoder (Upsampling)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Ensure tensor sizes match for concatenation
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=False)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        # Apply final layers to squeeze temporal dimension to 1
        x = self.final_conv_layers(x)  # Shape: (batch, channels, 1, H, W)
        return x.squeeze(2)  # Remove the temporal dimension

def test():
    num_input_frames = 16 # 16 is the MINIMUM NUMBER OF FRAMES
    x = torch.randn((3, 1, num_input_frames, 64, 64))  # (batch, channels, time, height, width)
    model = UNet3D(in_channels=1, out_channels=1, num_input_frames=num_input_frames)
    preds = model(x)
    
    assert preds.shape == (3, 1, 64, 64), f"Expected shape {(3, 1, 64, 64)}, but got {preds.shape}"

    print(preds.shape)

if __name__ == "__main__":
    test()
    print("Test complete")
