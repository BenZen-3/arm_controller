import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path


class VideoDataset(Dataset):
    """
    A dataset class for handling video recordings and preparing input-output pairs
    for training and testing deep learning models.
    """
    def __init__(self, device, recordings=None, num_frames=10, entry_point=None, data_path="data"):
        """
        Initializes the dataset by loading or processing video recordings.
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_frames = num_frames
        self.data_path = Path(data_path)
        self.data = []
        self.labels = []

        if not recordings:
            self.load_recordings(entry_point)

        self.prepare_data()

    def load_recordings(self, path):
        """
        Loads recordings from the specified path and stores them in memory.
        """
        data_path = Path(path) / self.data_path
        for file in data_path.iterdir():
            if file.suffix == ".npy":
                self.recordings.append(np.load(file))

    def prepare_data(self):
        """
        Processes the recordings into input-output pairs for training.
        Each sample consists of `num_frames` consecutive frames as input
        and the next frame as the label.
        """
        for rec in self.recordings:
            for i in range(len(rec) - self.num_frames):
                self.data.append(rec[i:i + self.num_frames])
                self.labels.append(rec[i + self.num_frames])

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
    A 3D convolutional neural network model for video frame prediction.
    """
    def __init__(self, input_channels=1, output_channels=1, num_frames=10):
        """
        Initializes the 3D CNN model architecture.
        """
        super().__init__()
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

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(2)


class VideoPredictor:
    """
    A class to handle future frame prediction using a trained VideoConv3D model.
    """
    def __init__(self, model, device):
        """
        Initializes the predictor with a trained model and device.
        """
        self.model = model.to(device)
        self.device = device

    def predict_future_frames(self, initial_frames, num_future_frames):
        """
        Generates future frames based on the initial input frames.
        """
        self.model.eval()
        input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        predicted_frames = []

        with torch.no_grad():
            for _ in range(num_future_frames):
                next_frame = self.model(input_frames).squeeze().cpu().numpy()
                predicted_frames.append(next_frame)
                input_frames = torch.cat(
                    (input_frames[:, :, 1:, :, :], torch.tensor(next_frame, dtype=torch.float32, device=self.device)
                     .unsqueeze(0).unsqueeze(0).unsqueeze(2)), dim=2)
        
        return np.array(predicted_frames)


class VideoTrainer:
    """
    A class for training and testing the VideoConv3D model.
    """
    def __init__(self, model, device, learning_rate=0.001):
        """
        Initializes the training setup with model, device, and optimizer.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataloader, num_epochs=10):
        """
        Trains the model using the given dataset and optimizer.
        """
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    def save_model(self, path):
        """Saves the trained model to a specified path."""
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load_model(path, input_channels=1, output_channels=1, device="cpu"):
        """Loads a saved model from disk."""
        model = VideoConv3D(input_channels, output_channels)
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)

    @staticmethod
    def test_model(model, test_loader, device):
        """Evaluates the model on a test dataset."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs.squeeze(1).cpu().numpy())

        return np.concatenate(predictions, axis=0)


if __name__ == "__main__":
    pass
