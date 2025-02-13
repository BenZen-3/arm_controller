from .simulator import Recording
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from . import utils
import numpy as np
from os import listdir
# from torch.cuda.amp import autocast, GradScaler

# TODO: switch away from a pre-allocated array because i have like no ram :(

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

        # num_samples = sum(len(rec.frame_sequence) - self.num_input_frames for rec in self.recordings) - 2*len(self.recordings)
        # print(num_samples)

        # Preallocate NumPy arrays
        data_list = [] #np.zeros((num_samples, self.num_input_frames, 82, 82), dtype=np.float32)  # TODO: HARDCODED 82
        label_list = [] #np.zeros((num_samples, self.num_label_frames, 82, 82), dtype=np.float32)

        index = 0
        for num, rec in enumerate(self.recordings):
            print(f"loaded {num+1}/{len(self.recordings)} recordings")
            frame_seq = rec.get_float_frame_seq()
            # print(np.shape(frame_seq))
            for i in range(len(frame_seq) - self.num_input_frames - self.num_label_frames):
                data = np.copy(frame_seq[i:i + self.num_input_frames])
                label = np.copy(frame_seq[i + self.num_input_frames : i + self.num_input_frames + self.num_label_frames])
                data_list.append(data)
                label_list.append(label)
                index += 1

        import time
        start = time.time()
        data_list = np.stack(data_list)
        label_list = np.stack(label_list)
        print(time.time() - start)

        self.data = torch.from_numpy(data_list)
        self.labels = torch.from_numpy(label_list)

        print(f"data size: {np.shape(data_list)}")
        print(f"labels size: {np.shape(label_list)}")

        return

                # PREV WORKING
                # data[index] = np.copy(frame_seq[i:i + self.num_input_frames])
                # labels[index] = np.copy(frame_seq[i + self.num_input_frames : i + self.num_input_frames + self.num_label_frames])
                # index += 1



                # OLD
                # print(index)
                # print(f"i:{i}")
                # print(f"{i + self.num_input_frames} : {i + self.num_input_frames + self.num_label_frames}")
                # print(f"input {self.num_input_frames}")
                # print(f"label {self.num_label_frames}")
                #np.ones((82,82))#np.copy(frame_seq[i + self.num_frames])#:i + 2*self.num_frames]) # 2*_ might be wrong
                # print(np.shape(labels[index]))

                # frame = labels[index]
                # print(np.shape(frame))
                # Recording.frame_printer(frame[0])

                # print(np.max(labels[index]))

        print("WHY SO SLOW")

        print(len(data_list))
        print(len(label_list))

        data_list = np.stack(data_list)
        label_list = np.stack(label_list)

        print(f"data size: {np.shape(data_list)}")
        print(f"labels size: {np.shape(label_list)}")

        # 24945
        # 24945
        # data size: (10, 82, 82)
        # labels size: (1, 82, 82)

        # return

        # data_list = np.concatenate(data_list)
        # label_list = np.concatenate(label_list)

        # SHOULD GENERATE (24950, 10, 82, 82)

        self.data = torch.from_numpy(data_list)
        self.labels = torch.from_numpy(label_list)

        print(f"data size: {np.shape(data_list)}")
        print(f"labels size: {np.shape(label_list)}")

        # for frame_seq in self.data:
        #     # print(np.shape(item))
        #     print(np.shape(frame_seq))
        #     for frame in frame_seq:
        #         frame = np.array(frame.cpu().numpy())
        #         print(np.shape(frame))
        #         Recording.frame_printer(frame)

        # first_data = data[0]
        # first_label = labels[0]

        # self.view_data_set(1)

    # def view_data_set(self, num=0):
    #     """
    #     view the dataset based on data point number. 
    #     """

    #     # TODO: REMOVE THIS 

    #     data_seq, label = self.__getitem__(num)
    #     # label = self.labels[num]
    #     print(type(data_seq))
    #     print(type(label))
    #     print(np.shape(data_seq)) # torch.Size([1, 10, 82, 82])
    #     print(np.shape(label))    # torch.Size([1, 2, 82, 82])
    #     import time
    #     # for seq in data_seq:
    #     for frame in data_seq[0]:
            
    #         print('tf')
    #         frame = np.array(frame.cpu().numpy())
    #         print(np.max(frame))
    #         # print(f"HERE DIKBD:SHDNFSL:FSJL:FJ: {np.shape(frame)}")
    #         Recording.frame_printer(frame)
    #         time.sleep(.1)
        

    #     # for l in label:
    #     #     print(l)
    #     #     l = np.array(l.cpu().numpy())
    #     #     print(f"HERE DIKBD:SHDNFSL:FSJL:FJ: {np.shape(l)}")
    #     #     Recording.frame_printer(l)
    #     #     time.sleep(.1)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the sample at the given index.""" 
        x = self.data[idx].unsqueeze(0).to(self.device, non_blocking=True) # DataPoint x Channels x frame_count x H x W
        y = self.labels[idx].unsqueeze(0).to(self.device, non_blocking=True)
        # x = self.data[idx].to(self.device, non_blocking=True)
        # y = self.labels[idx].to(self.device, non_blocking=True)

        return x, y

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        weight_mask = torch.where(target > 0.1, self.weight, 1.0)  # Increase weight for bright areas
        loss = self.mse(pred * weight_mask, target * weight_mask)
        return loss


class RecursivePredictionLoss(nn.Module):
    def __init__(self, prediction_func, device = 'cuda', num_frames=10):
        super().__init__()
        self.prediction_func = prediction_func
        self.device = device
        self.num_frames = num_frames
        self.core_loss_func = nn.MSELoss()

    def forward(self, inputs, target):
        """
        custom loss function for predicting frames using its own generated frames
        """
        # inputs and outputs have the same size here: torch.Size([32, 1, 10, 82, 82])

        predicted_frames = self.prediction_func(inputs, self.num_frames) # TODO: This HAS to be a 10 for now
        # predicted_frames = torch.tensor(predicted_frames, dtype=torch.float32, device=self.device)#.unsqueeze(0).unsqueeze(0)
        # predicted_frames = predicted_frames.clone().detach().requires_grad_(True)

        # print(f"predicted frame shape: {np.shape(predicted_frames)}")
        # print(f"target shape: {np.shape(target)}")

        # loss = torch.mean((predicted_frames - target)**2)

        return self.core_loss_func(predicted_frames, target)



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
            nn.Conv3d(a, output_channels, kernel_size=(num_frames, 3, 3), stride=(num_frames, 1, 1), padding=(0, 1, 1)),
        )

        # self.loss_fn = RecursivePredictionLoss(self.predict_future_frames)
        self.loss_fn = nn.MSELoss()#WeightedMSELoss(weight=10.0)  # Adjust weight if needed
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

                # print(f"inputs: {np.shape(inputs)}")
                # print(f"targets: {np.shape(targets)}")
               
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                #     # inputs, targets = inputs, targets.unsqueeze(1)  # Add channel dimension to targets
                #     inputs = inputs.to(self.device, non_blocking=True)
                #     targets = targets.to(self.device, non_blocking=True)  # THIS MIGHT REQUIRE AN UNSQUEEZE WHEN THERE ARE MULTIPLE OUTPUT FRAMES

                #     outputs = self(inputs)
                #     loss = self.loss_fn(outputs, targets)

                #     # print(f"inputs: {np.shape(inputs)}") # ([32, 1, 10, 82, 82])
                #     # print(f"targets: {np.shape(targets)}") # ([32, 1, ?1?, 82, 82])
                #     # print(f"outputs: {np.shape(outputs)}") # ([32, 1, 1, 82, 82])
                #     # print(f"Loss: {loss}")

                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                # self.optimizer.zero_grad()

                self.optimizer.zero_grad()

                inputs, targets = inputs, targets

                print(f"inputs: {np.shape(inputs)}")
                print(f"targets: {np.shape(targets)}")

                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # garbo past here
                percent = round(batch_num / len(dataloader) * 100)
                if percent != old_percent:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Progress: {percent}%, Loss: {running_loss/(batch_num+1)}")
                    old_percent = percent

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.8f}")

    def predict_future_frames(self, initial_frames, num_future_frames):
        """
        Generates future frames recursively while maintaining differentiability.
        Uses the model's own predictions as input for future time steps.
        """
        # self.eval()  # Ensure evaluation mode TODO: NEED ANOTHER PREDICT FOR TESTING

        # # Ensure input tensor has the correct shape: (batch_size, 1, num_frames, 82, 82)
        # if isinstance(initial_frames, np.ndarray):
        #     input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        #     print("WHAT THE FUCK")
        # else:
        #     input_frames = initial_frames  # Assume correct shape (batch_size, 1, num_frames, 82, 82)

        input_frames = initial_frames  # Assume correct shape (batch_size, 1, num_frames, 82, 82)
        predicted_frames = []

        with torch.set_grad_enabled(self.training):  # Enable gradients only during training
            for _ in range(num_future_frames):
                # Forward pass, ensuring correct output shape (batch_size, 1, 1, 82, 82)
                next_frame = self(input_frames)  # Model prediction

                # print(f"next frame shape: {np.shape(next_frame)}")

                # Append new frame to predictions
                predicted_frames.append(next_frame.unsqueeze(2))  # Keep as tensor

                # Prepare next input: remove first frame, append new frame
                input_frames = torch.cat((input_frames[:, :, 1:, :, :], next_frame.unsqueeze(2)), dim=2)  # Maintain correct shape

        # Stack along the time dimension to form (batch_size, 1, num_future_frames, 82, 82)
        recursive_prediction = torch.cat(predicted_frames, dim=2)

        return recursive_prediction  # Fully torch-based, maintains computation graph


    # def predict_future_frames(self, initial_frames, num_future_frames):
    #     """
    #     Generates future frames based on the initial input frames.
    #     Ensures correct batch and channel shape for batch processing.
    #     """
    #     self.eval()

    #     # Ensure input tensor has the correct shape: (batch_size, channels=1, num_frames, height=82, width=82)
    #     if isinstance(initial_frames, np.ndarray):
    #         input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
    #     else:
    #         input_frames = initial_frames  # Assume correct shape (batch_size, 1, num_frames, 82, 82)

    #     predicted_frames = []

    #     with torch.no_grad():
    #         for _ in range(num_future_frames):
    #             # Forward pass, ensuring correct output shape (batch_size, 1, 1, 82, 82)
    #             next_frame = self(input_frames)  # Expected output: (batch_size, 1, 1, 82, 82)

    #             # Prepare next input (remove first frame, append new frame)
    #             prior_inputs = input_frames[:, :, 1:, :, :]  # Keep batch, remove oldest frame
    #             output = next_frame[:, :, :, :].unsqueeze(2)  # Ensure it has shape (batch_size, 1, 1, 82, 82)

    #             predicted_frames.append(output.cpu().numpy())
    #             input_frames = torch.cat((prior_inputs, output), dim=2)  # Shape: (batch_size, 1, num_frames, 82, 82)

    #     recursive_prediction = np.stack(predicted_frames, axis=2).squeeze(3)
    #     return recursive_prediction

    def predict_future_frames_testing(self, initial_frames, num_future_frames):
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
    dataset = VideoDataset(device, num_input_frames=10,num_label_frames=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VideoConv3D(device=device)
    print('started training process')
    model.train_model(dataloader, num_epochs=1)

    model_save_path = utils.get_model_folder()
    model.save_model(model_save_path)


def main_predict(seed_frames, num_future_frames=10):
    """
    Loads a trained model and generates future frames given a seed frame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = utils.get_most_recent_model()
    print(f"Running model found at {model_path}")

    model = VideoConv3D.load_model(model_path, device=device)
    future_frames = model.predict_future_frames_testing(seed_frames, num_future_frames)
    return future_frames
