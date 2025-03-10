from .simulator import Recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from . import utils
import numpy as np
from os import listdir
import time # REMOVE LATER

# TODO find better name for num_label_frames

class VideoDataset(Dataset):
    """
    A dataset class for handling video recordings and preparing input-output pairs
    for training and testing deep learning models.
    """
    def __init__(self, device, num_input_frames, num_label_frames, recordings=None):
        """
        Initializes the dataset by loading or processing video recordings.
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_input_frames = num_input_frames
        self.num_label_frames = num_label_frames
        self.data = None
        self.labels = None
        self.max_recordings = 5

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

                if num+1 > self.max_recordings: break # TODO: get more ram for my laptop...

    def prepare_data(self):
        """
        Processes the recordings into input-output pairs for training.
        Each sample consists of `num_frames` consecutive frames as input
        and the next frame as the label.
        """
        
        total_frames = sum(len(rec.frame_sequence) - self.num_input_frames - self.num_label_frames 
                        for rec in self.recordings)
        
        # Get input-output shape
        example_frame = self.recordings[0].frame_sequence[0]
        frame_shape = example_frame.shape  # Assuming frames are NumPy arrays
        
        # Preallocate arrays
        data_array = np.empty((total_frames, self.num_input_frames, *frame_shape), dtype=np.uint8)
        label_array = np.empty((total_frames, self.num_label_frames, *frame_shape), dtype=np.uint8)

        index = 0
        for num, rec in enumerate(self.recordings):
            # print(f"loaded {num+1}/{len(self.recordings)} recordings")
            frame_seq = rec.frame_sequence
            seq_len = len(frame_seq)

            for i in range(seq_len - self.num_input_frames - self.num_label_frames):
                data_array[index] = frame_seq[i:i + self.num_input_frames]
                label_array[index] = frame_seq[i + self.num_input_frames : i + self.num_input_frames + self.num_label_frames]
                index += 1

        self.data = torch.from_numpy(data_array)
        self.labels = torch.from_numpy(label_array)

        if index not in data_array.shape and index not in label_array.shape:
            print(f"Index suggests size should be: {index}")
            print(f"data array size: {data_array.shape}")
            print(f"labels array size: {label_array.shape}")
            raise Exception("index suggests that array size is wrong")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the sample at the given index.""" 
        x = self.data[idx].to(torch.float32).unsqueeze(0).to(self.device, non_blocking=True) / 255.0 # TODO: get this number from somewhere instead of this hardcoded bs
        y = self.labels[idx].to(torch.float32).unsqueeze(0).to(self.device, non_blocking=True) / 255.0

        return x, y # (Batch DataPoint x Channels x frame_count x H x W) , (Label)


class RecursivePredictionLoss(nn.Module):
    def __init__(self, prediction_func, device = 'cuda', num_input_frames=10, num_label_frames=1):
        super().__init__()
        self.prediction_func = prediction_func
        self.device = device
        self.num_input_frames = num_input_frames
        self.num_label_frames = num_label_frames
        self.core_loss_func = nn.MSELoss()

    def forward(self, inputs, target):
        """
        custom loss function for predicting frames using its own generated frames
        """

        predicted_frames = self.prediction_func(inputs, self.num_label_frames)
        return self.core_loss_func(predicted_frames, target)


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


class FramePredictionModel(nn.Module):

    def __init__(self, num_input_frames, num_label_frames, features=[32, 64, 128, 256], device="cpu", learning_rate=0.001):
        super().__init__()

        in_channels, out_channels = 1, 1

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

        self.device = device
        self.num_input_frames = num_input_frames
        self.num_label_frames = num_label_frames
        self.loss_fn = RecursivePredictionLoss(self.predict_future_frames, num_input_frames=self.num_input_frames, num_label_frames=self.num_label_frames)
        # self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

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

    def train_model(self, dataloader, num_epochs=10):
        """
        Trains the model using the given dataset and optimizer.
        """

        # scaler = torch.amp.GradScaler("cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=True)


        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            old_percent = -1

            for batch_num, (inputs, targets) in enumerate(dataloader):

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    loss = self.loss_fn(inputs, targets) # MODIFIED

                    # print(f"input size: {np.shape(inputs)}")
                    # print(f"target size: {np.shape(targets)}")
                    # print(f"output size: {np.shape(output)}")

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                running_loss += loss.item()

                # garbo past here
                percent = round(batch_num / len(dataloader) * 100)
                if percent != old_percent:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Progress: {percent}%, Loss: {running_loss/(batch_num+1)}")
                    old_percent = percent

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.8f}")

    def predict_future_frames(self, input_frames, num_future_frames, with_grad=True):
        """
        Generates future frames recursively.
        Uses the model's own predictions as input for future time steps.
        """

        predicted_frames = []

        with torch.set_grad_enabled(with_grad):  # Enable gradients only during training
            for _ in range(num_future_frames):
                next_frame = self(input_frames)  # Model prediction
                predicted_frames.append(next_frame.unsqueeze(2))

                # Prepare next input: remove first frame, concatenate new frame
                input_frames = torch.cat((input_frames[:, :, 1:, :, :], next_frame.unsqueeze(2)), dim=2)  # Maintain correct shape

        # Stack along the time dimension to form (batch_size, 1, num_future_frames, 82, 82)
        recursive_prediction = torch.cat(predicted_frames, dim=2)

        return recursive_prediction  # Fully torch-based, maintains computation graph

    def predict_future_frames_testing(self, initial_frames, num_future_frames):
        """
        predict future frames, but for testing the model instead of training
        """

        self.eval()

        # get frames in correct shape
        initial_frames = initial_frames[:self.num_input_frames]
        input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # get prediction and return it to normal memory
        prediction = self.predict_future_frames(input_frames, num_future_frames, with_grad=False).squeeze(0).squeeze(0)
        print(np.shape(prediction))
        return prediction.cpu().numpy()

    def save_model(self, save_folder):
        """Saves the trained model to a specified path."""

        id = len(listdir(save_folder)) # a bit jank considering it also counts the .gitignore
        name = f"frame_prediction_model_{id}.pth"
        save_file = save_folder.joinpath(name)
        print(f"Saved Model as {name}")

        torch.save(self.state_dict(), save_file)

    @staticmethod
    def load_model(path, num_input_frames=10, num_label_frames=1, device="cpu"):
        """Loads a saved model from disk."""
        model = FramePredictionModel(num_input_frames=num_input_frames, num_label_frames=num_label_frames, device=device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.to(device)

def main_train(use_stored_model=None):
    """
    Builds the model, prepares the dataset, and trains the model.
    """

    _num_input_frames = 16
    _num_label_frames = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoDataset(device, num_input_frames=_num_input_frames, num_label_frames=_num_label_frames)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    if use_stored_model:
        model_path = utils.get_most_recent_model()
        print(f"Training model found at {model_path}")
        model = FramePredictionModel.load_model(model_path, num_input_frames = _num_input_frames, num_label_frames=_num_label_frames, device=device)
    else: 
        model = FramePredictionModel(device=device, num_input_frames=_num_input_frames, num_label_frames=_num_label_frames)


    # TODO: This is user I/O. probably best to move it all to the main file
    try: 
        print('Started training process')
        model.train_model(dataloader, num_epochs=1)
    except KeyboardInterrupt:
        do_save = input(f"\n\nTraining stopped!\n\n    Would you like to save the model anyways? \"SAVE\" to save; Enter to discard: ") == "SAVE"
        if not do_save:
            return

    model_save_path = utils.get_model_folder()
    model.save_model(model_save_path)


def main_predict(seed_frames, num_future_frames=10):
    """
    Loads a trained model and generates future frames given a seed frame.
    """

    _num_input_frames = 16
    _num_label_frames = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = utils.get_most_recent_model()
    print(f"Running model found at {model_path}")

    model = FramePredictionModel.load_model(model_path, num_input_frames = _num_input_frames, num_label_frames=_num_label_frames, device=device)
    future_frames = model.predict_future_frames_testing(seed_frames, num_future_frames)
    return future_frames
