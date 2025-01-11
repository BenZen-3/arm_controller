import torch
import torch.nn as nn
import numpy as np
# from pathlib import Path


class VideoConv3D(nn.Module):

    data_path = "data"

    def __init__(self, input_channels=1, output_channels=1):
        super(VideoConv3D, self).__init__()
        self.recordings = []
        
        # 3D Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, output_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
        )

    def forward(self, x):
        # Pass through encoder
        x = self.encoder(x)
        # Pass through decoder
        x = self.decoder(x)
        return x

    def load_recordings(self, path):

        data_path = path.joinpath(f"{self.data_path}")

        for sim_data_path in data_path.iterdir():
            if sim_data_path.is_file() and sim_data_path.suffix == ".npy":
                # print(sim_data_path)

                self.recordings.append(np.load(sim_data_path))
                print(np.shape(self.recordings))
    

# Example usage
def main():
    # Video tensor shape: (Batch Size, Channels, Time Frames, Height, Width)
    batch_size = 2
    input_channels = 1
    time_frames = 10
    height, width = 82, 82

    # Create a random input tensor
    video_input = torch.rand(batch_size, input_channels, time_frames, height, width)

    # Instantiate the model
    model = VideoConv3D(input_channels=input_channels, output_channels=1)

    # Forward pass
    output = model(video_input)

    # Print the shapes
    print("Input shape:", video_input.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()