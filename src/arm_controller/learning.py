from .simulator import Recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, device, num_input_frames, num_label_frames=10, recordings=None):
        """
        Initializes the dataset by loading or processing video recordings.
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_input_frames = num_input_frames
        self.num_label_frames = num_label_frames
        self.data = None
        self.labels = None
        self.max_recordings = 200

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
        
        total_frames = sum(len(rec.get_float_frame_seq()) - self.num_input_frames - self.num_label_frames 
                        for rec in self.recordings)
        
        # Get input-output shape
        example_frame = self.recordings[0].get_float_frame_seq()[0]
        frame_shape = example_frame.shape  # Assuming frames are NumPy arrays
        
        # Preallocate arrays
        data_array = np.empty((total_frames, self.num_input_frames, *frame_shape), dtype=np.float32)
        label_array = np.empty((total_frames, self.num_label_frames, *frame_shape), dtype=np.float32)

        index = 0
        for num, rec in enumerate(self.recordings):
            # print(f"loaded {num+1}/{len(self.recordings)} recordings")
            frame_seq = rec.get_float_frame_seq() # TODO: THIS IS A MEMORY EATING HUNK OF JUNK. DO DIVISION LATER
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
        
        print(np.shape(self.recordings[0].get_float_frame_seq()))
        print(len(self.data))
        print(len(self.labels))
        print(f"data array size: {data_array.shape}")
        print(f"labels array size: {label_array.shape}")

        # for frame in np.array(self.data[0].cpu().numpy()):
        #     Recording.frame_printer(frame)

        # time.sleep(10)


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the sample at the given index.""" 
        x = self.data[idx].unsqueeze(0).to(self.device, non_blocking=True) # DataPoint x Channels x frame_count x H x W
        y = self.labels[idx].unsqueeze(0).to(self.device, non_blocking=True)

        return x, y

class WeightedMSELossOLD(nn.Module):
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        weight_mask = torch.where(target > 0.1, self.weight, 1.0)  # Increase weight for bright areas
        loss = self.mse(pred * weight_mask, target * weight_mask)
        return loss


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


        # class FramePredictionModel(nn.Module):
        #     """
        #     Neural network for video frame prediction,
        #     including training and prediction functionalities.
        #     """
        #     def __init__(self, patch_size=16, num_input_frames=10, num_label_frames=1, dim=256, num_heads=4, num_layers=4, device="cpu", learning_rate=0.001):
        #         """
        #         Initializes the model architecture along with training utilities.
        #         """
        #         super().__init__()
        #         self.device = device

        #         self.frame_size = 64 # TODO: not hardcode

        #         self.cnn_encoder = nn.Sequential(
        #             nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=1),
        #             nn.ReLU(),
        #             nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
        #             nn.ReLU(),
        #         )
        #         self.patch_embed = nn.Conv3d(64, dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        #         self.transformer = nn.Transformer(dim, num_heads, num_layers, batch_first=True)
        #         self.decoder = nn.Sequential(
        #             nn.ConvTranspose3d(dim, 64, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size)),
        #             nn.ReLU(),
        #             nn.Conv3d(64, 1, kernel_size=(3,3,3), padding=1)
        #         )

        #         self.loss_fn = RecursivePredictionLoss(self.predict_future_frames, num_input_frames=num_input_frames, num_label_frames=num_label_frames)
        #         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #         self.to(device)

        #     def forward(self, x):
        #         """Defines the forward pass of the model."""
        #         x = self.cnn_encoder(x)  # CNN for spatial feature extraction
        #         x = self.patch_embed(x)  # Convert feature maps into patches
        #         x = x.flatten(2).permute(2, 0, 1)  # Reshape for Transformer
        #         x = self.transformer(x, x)  # Transformer for temporal dependencies
        #         # x = x.permute(1, 2, 0).reshape(x.shape[1], 256, 10, self.frame_size, self.frame_size)  # Ensure 256 channels
        #         # return self.decoder(x)  # Reconstruct frame

        #         # print(f"Shape after transformer: {x.shape}")  # Debugging step
        #         # batch_size, seq_len, features = x.shape  # Extract actual dimensions
        #         # x = x.reshape(batch_size, features, 10, self.frame_size, self.frame_size)  # Adjust dynamically

        #         batch_size, seq_len, features = x.shape  # Extract dimensions
        #         x = x.permute(1, 2, 0).reshape(batch_size, features, seq_len, 1, 1)  # Reshape based on actual size

        #         x = self.decoder(x)

        #         print(f"Shape after decoder: {x.shape}")  # Shape after decoder: torch.Size([160, 1, 8, 16, 16])


        #         return x


class FramePredictionModel(nn.Module):
    """
    A 3D convolutional neural network model for video frame prediction,
    including training and prediction functionalities.
    """
    def __init__(self, input_channels=1, output_channels=1, num_input_frames=10, num_label_frames=1, device="cpu", learning_rate=0.001):
        """
        Initializes the 3D CNN model architecture along with training utilities.
        """
        super().__init__()
        self.device = device

        self.num_input_frames = num_input_frames
        self.num_label_frames = num_label_frames

        a = 16
        b = 32

        # 3D Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, a, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(a, b, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(b, a, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(a, output_channels, kernel_size=(num_input_frames, 3, 3), stride=(num_input_frames, 1, 1), padding=(0, 1, 1)),
        )


        # a, b = 128, 256
        
        # # Encoder with residual blocks
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(input_channels, a, kernel_size=(3,3,3), stride=1, padding=1),
        #     nn.ReLU(),
        #     # Residual3DBlock(a),
        #     nn.Conv3d(a, b, kernel_size=(3,3,3), stride=1, padding=1),
        #     nn.ReLU(),
        #     # Residual3DBlock(b)
        # )
        
        # # # Bottleneck: 
        # # self.bottleneck = nn.Sequential(
        # #     nn.Conv3d(b, b, kernel_size=(3,3,3), stride=1, padding=1),  # Smaller kernel
        # #     nn.ReLU(),
        # # )

        # # Decoder with upsampling
        # self.decoder = nn.Sequential(
        #     # Residual3DBlock(b),
        #     nn.Conv3d(b, a, kernel_size=(3,3,3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(a, a, kernel_size=(3,3,3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(a, output_channels, kernel_size=(num_input_frames, 3, 3), stride=(num_input_frames, 1, 1), padding=(0, 1, 1))
        # )

        self.loss_fn = RecursivePredictionLoss(self.predict_future_frames, num_input_frames=self.num_input_frames, num_label_frames=self.num_label_frames)
        # self.loss_fn = nn.MSELoss()
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

        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            old_percent = -1

            for batch_num, (inputs, targets) in enumerate(dataloader):

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    loss = self.loss_fn(inputs, targets) # MODIFIED

                    print(np.shape(inputs))
                    print(np.shape(targets))
                    print(np.shape(self(inputs)))

                    # I/O here:
                    # torch.Size([10, 1, 10, 64, 64])
                    # torch.Size([10, 1, 64, 64])



                    # for f in inputs[0][0].cpu().numpy():
                    #     Recording.frame_printer(f)
                    #     time.sleep(.01)


                    # Recording.frame_printer(targets[0][0][0].cpu().numpy())

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


    def predict_future_frames_testingOLD(self, initial_frames, num_future_frames):
        """
        Generates future frames based on the initial input frames.
        """

        # I/O here:
        # torch.Size([1, 1, 21, 64, 64])
        # torch.Size([1, 1, 2, 64, 64])

        print(np.shape(initial_frames))
        self.eval()
        initial_frames = initial_frames[:self.num_input_frames]
        input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        print(np.shape(input_frames))
        predicted_frames = []

        with torch.no_grad():
            for _ in range(num_future_frames):
                next_frame = self(input_frames).squeeze().cpu().numpy()
                print(np.shape(self(input_frames)))

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
    def load_model(path, num_input_frames=10, num_label_frames=1, device="cpu"):
        """Loads a saved model from disk."""
        model = FramePredictionModel(num_input_frames=num_input_frames, num_label_frames=num_label_frames, device=device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.to(device)

def main_train(use_stored_model=None):
    """
    Builds the model, prepares the dataset, and trains the model.
    """

    _num_input_frames = 10
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
        model.train_model(dataloader, num_epochs=10000)
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

    _num_input_frames = 10
    _num_label_frames = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = utils.get_most_recent_model()
    print(f"Running model found at {model_path}")

    model = FramePredictionModel.load_model(model_path, num_input_frames = _num_input_frames, num_label_frames=_num_label_frames, device=device)
    future_frames = model.predict_future_frames_testing(seed_frames, num_future_frames)
    return future_frames
