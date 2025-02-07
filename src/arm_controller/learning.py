from .simulator import Recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from . import utils
import numpy as np
from os import listdir
from torch.cuda.amp import autocast, GradScaler



class VideoDataset(Dataset):
    """
    A dataset class for handling video recordings and preparing input-output pairs
    for training and testing deep learning models.
    """
    def __init__(self, device, num_frames, recordings=None):
        """
        Initializes the dataset by loading or processing video recordings.
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_frames = num_frames
        self.data = []
        self.labels = []
        self.max_recordings = 250

        if not recordings:
            self.load_recordings()

        self.prepare_data()

    def load_recordings(self):
        """
        Loads recordings from the specified path and stores them in memory.
        """
        for num, file in enumerate(utils.get_data_folder().iterdir()):
            if file.suffix == ".npz":
                rec = Recording()
                rec.init_from_file(file)
                self.recordings.append(rec)

                if num > self.max_recordings: break # TODO: get more ram for my laptop...

    def prepare_data(self):
        """
        Processes the recordings into input-output pairs for training.
        Each sample consists of `num_frames` consecutive frames as input
        and the next frame as the label.
        """
        for num, rec in enumerate(self.recordings):
            print(f"loaded {num+1}/{len(self.recordings)} recordings")
            frame_seq = rec.get_float_frame_seq()
            # frame_seq = rec.frame_sequence.astype(np.float32) / 255.0 # TODO: Check the datatype and divide by datatype max instead...
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
        self.criterion = nn.L1Loss() # self.criterion = nn.MSELoss()
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

        print(len(dataloader))

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            counter = 0 # switch to enumerate probs
            old_percent = -1

            for inputs, targets in dataloader:
                self.optimizer.zero_grad()

                with torch.autocast(device_type="cuda"):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets.unsqueeze(1))

                # outputs = self(inputs)
                # loss = self.criterion(outputs, targets.unsqueeze(1))


                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # garbo past here
                counter += 1
                percent = round(counter / len(dataloader) * 100)
                if percent != old_percent:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Progress: {percent}%, Loss: {running_loss/counter}")
                    old_percent = percent

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.8f}")

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

    def save_model(self, save_folder):
        """Saves the trained model to a specified path."""

        id = len(listdir(save_folder)) # a bit jank considering it also counts the .gitignore
        name = f"frame_prediction_model_{id}.pth"
        save_file = save_folder.joinpath(name)
        print(f"Saved Model as {name}")

        torch.save(self.state_dict(), save_file)

    @staticmethod
    def load_model(path, input_channels=1, output_channels=1, num_frames=10, device="cpu"):
        """Loads a saved model from disk."""
        model = VideoConv3D(input_channels, output_channels, num_frames, device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.to(device)


def main_train():
    """
    Builds the model, prepares the dataset, and trains the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDataset(device, num_frames=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VideoConv3D(device=device)
    print('started training process')
    model.train_model(dataloader, num_epochs=2)

    model_save_path = utils.get_model_folder()
    model.save_model(model_save_path)


def main_predict(seed_frames, num_future_frames=30):
    """
    Loads a trained model and generates future frames given a seed frame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = utils.get_most_recent_model()
    print(model_path)

    model = VideoConv3D.load_model(model_path, device=device)
    future_frames = model.predict_future_frames(seed_frames, num_future_frames)
    return future_frames


if __name__ == "__main__":
    main_train()
