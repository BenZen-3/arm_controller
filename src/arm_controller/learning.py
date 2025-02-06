from .simulator import Recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from . import utils
import numpy as np


class VideoDataset(Dataset):
    """
    A dataset class for handling video recordings and preparing input-output pairs
    for training and testing deep learning models.
    """
    def __init__(self, device, recordings=None, num_frames=10):
        """
        Initializes the dataset by loading or processing video recordings.
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_frames = num_frames
        self.data = []
        self.labels = []

        if not recordings:
            self.load_recordings()

        self.prepare_data()

    def load_recordings(self):
        """
        Loads recordings from the specified path and stores them in memory.
        """
        for file in utils.get_data_folder().iterdir():
            if file.suffix == ".npz":
                rec = Recording()
                rec.init_from_file(file)
                self.recordings.append(rec)

    def prepare_data(self):
        """
        Processes the recordings into input-output pairs for training.
        Each sample consists of `num_frames` consecutive frames as input
        and the next frame as the label.
        """
        for rec in self.recordings:
            frame_seq = rec.frame_sequence / 255.0 # TODO: Check the datatype and divide by datatype max instead...
            for i in range(len(frame_seq) - self.num_frames):
                self.data.append(frame_seq[i:i + self.num_frames])
                self.labels.append(frame_seq[i + self.num_frames])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the sample at the given index."""
        x = torch.tensor(self.data[idx], dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)
        return x, y


class VideoConv3D(nn.Module):
    """
    A 3D convolutional neural network model for video frame prediction,
    including training and prediction functionalities.
    """
    def __init__(self, input_channels=1, output_channels=1, num_frames=10, device="cpu", learning_rate=0.001):
        """
        Initializes the 3D CNN model architecture along with training utilities.
        """
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, output_channels, kernel_size=(num_frames, 3, 3),
                      stride=(num_frames, 1, 1), padding=(0, 1, 1))
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(2)

    def train_model(self, dataloader, num_epochs=10):
        """
        Trains the model using the given dataset and optimizer.
        """
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    def predict_future_frames(self, initial_frames, num_future_frames):
        """
        Generates future frames based on the initial input frames.
        """
        self.eval()
        input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        predicted_frames = []

        with torch.no_grad():
            for _ in range(num_future_frames):
                next_frame = self(input_frames).squeeze().cpu().numpy()
                predicted_frames.append(next_frame)
                input_frames = torch.cat(
                    (input_frames[:, :, 1:, :, :], torch.tensor(next_frame, dtype=torch.float32, device=self.device)
                     .unsqueeze(0).unsqueeze(0).unsqueeze(2)), dim=2)
        return np.array(predicted_frames)

    def save_model(self, path):
        """Saves the trained model to a specified path."""
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, input_channels=1, output_channels=1, num_frames=10, device="cpu"):
        """Loads a saved model from disk."""
        model = VideoConv3D(input_channels, output_channels, num_frames, device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)


def main_train():
    """
    Builds the model, prepares the dataset, and trains the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDataset(device, num_frames=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VideoConv3D(device=device)
    print('started training process')
    model.train_model(dataloader, num_epochs=1)

    model_save_path = utils.get_model_folder()
    model.save_model("video_conv3d.pth", model_save_path)


if __name__ == "__main__":
    main_train()
