import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class VideoDataset(Dataset):

    def __init__(self, device, recordings=None, num_frames=10, entry_point=None, data_path = "data"):
        """
        recordings: List of numpy arrays, each with shape (n, 82, 82) or None
        num_frames: Number of frames to use as input for prediction
        data_path: Path to the folder containing .npy files
        """
        self.device = device
        self.recordings = recordings if recordings is not None else []
        self.num_frames = num_frames
        self.data = []
        self.labels = []
        self.data_path = data_path

        # Load recordings if none are provided
        if not recordings:
            self.load_recordings(entry_point)

        # Prepare data and labels
        for rec in self.recordings:
            for i in range(len(rec) - num_frames):
                self.data.append(rec[i:i + num_frames])
                self.labels.append(rec[i + num_frames])


    def load_recordings(self, path):
        """
        Load .npy files from the specified directory.
        """
        data_path = path.joinpath(f"{self.data_path}")
        for sim_data_path in data_path.iterdir():
            if sim_data_path.is_file() and sim_data_path.suffix == ".npy":
                self.recordings.append(np.load(sim_data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        # y = torch.tensor(self.labels[idx], dtype=torch.float32)  # Single frame as label
        x = torch.tensor(self.data[idx], dtype=torch.float32, device=self.device).unsqueeze(0)  # Add channel dimension
        y = torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)  # Single frame as label

        return x, y
    

class VideoConv3D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_frames=10):
        super(VideoConv3D, self).__init__()
        
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
            nn.Conv3d(16, output_channels, kernel_size=(num_frames, 3, 3), stride=(num_frames, 1, 1), padding=(0, 1, 1)),
        )

    def forward(self, x):
        # Pass through encoder
        x = self.encoder(x)
        # Pass through decoder
        x = self.decoder(x)
        return x.squeeze(2)  # Remove the temporal dimension (T=1)


def predict_future_framesNEW(model, initial_frames, num_future_frames, device):
    """
    Predict a sequence of future frames given an initial set of frames.

    Parameters:
        model (nn.Module): The trained model.
        initial_frames (np.ndarray): Initial frames of shape (num_frames, 82, 82).
        num_future_frames (int): Number of future frames to predict.
        device (torch.device): The device to run the model on.

    Returns:
        np.ndarray: Predicted future frames of shape (num_future_frames, 82, 82).
    """
    model.eval()  # Set model to evaluation mode
    # num_frames = initial_frames.shape[0]  # Number of input frames

    # Prepare initial input tensor
    input_frames = torch.tensor(initial_frames, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # Shape: (1, 1, num_frames, 82, 82)

    predicted_frames = initial_frames

    with torch.no_grad():
        for _ in range(num_future_frames):
            # Predict the next frame
            output_frame = model(input_frames)  # Output shape: (1, 1, 82, 82)
            next_frame = output_frame.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (82, 82)

            predicted_frames.append(next_frame)

            # Update input by removing the oldest frame and appending the new one
            next_frame_tensor = torch.tensor(next_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(2)
            input_frames = torch.cat((input_frames[:, :, 1:, :, :], next_frame_tensor), dim=2)

    return np.array(predicted_frames)  # Shape: (num_future_frames, 82, 82)



def predict_future_frames(model, initial_frames, num_future_frames, device):
    """
    Predict a sequence of future frames given an initial set of frames.

    Parameters:
        model (nn.Module): The trained model.
        initial_frames (np.ndarray): Initial frames of shape (num_frames, 82, 82).
        num_future_frames (int): Number of future frames to predict.
        device (torch.device): The device to run the model on.

    Returns:
        np.ndarray: Predicted future frames of shape (num_future_frames, 82, 82).
    """
    model.eval()
    initial_frames = torch.tensor(initial_frames, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_frames, 82, 82)
    predicted_frames = []

    with torch.no_grad():
        for _ in range(num_future_frames):
            # Predict the next frame
            next_frame = model(initial_frames)
            next_frame = next_frame.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (82, 82)
            predicted_frames.append(next_frame)

            # Update the input by removing the oldest frame and adding the predicted frame
            initial_frames = torch.cat((initial_frames[:, :, 1:, :, :], torch.tensor(next_frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(2)), dim=2)

    return np.array(predicted_frames)

def train_model(model, dataloader, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        model.to(device)

        counter = 0 # switch to enumerate probs
        old_percent = -1
        for inputs, targets in dataloader:
            # Move data to device if applicable
            inputs, targets = inputs, targets.unsqueeze(1)  # Add channel dimension to targets

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(running_loss)

            # garbo past here
            counter += 1
            percent = round(counter / len(dataloader) * 100)
            if percent != old_percent:
                print(f"Progress: {percent}%")
                old_percent = percent

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(path, input_channels=1, output_channels=1, device="cpu"):
    """Load the model from the specified path."""
    model = VideoConv3D(input_channels=input_channels, output_channels=output_channels)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    return model

def test_model(model, test_loader, device):
    """Run inference on the test dataset."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        old_percent = -1
        counter = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Shape: (batch_size, 1, 82, 82)
            predictions.append(outputs.squeeze(1).cpu().numpy())  # Remove channel dim: (batch_size, 82, 82)
            
            # garbo past here
            counter += 1
            percent = round(counter / len(test_loader) * 100)
            if percent != old_percent:
                print(f"Progress: {percent}%")
                old_percent = percent
    
    # Flatten the predictions into a single array
    predictions = np.concatenate(predictions, axis=0)  # Shape: (total_frames, 82, 82)
    return predictions

def train(entry_point):

    # Example data setup
    print('training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device=}")

    dataset = VideoDataset(device, num_frames=10, entry_point=entry_point)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print('made dataset and loader')

    # Instantiate the model
    model = VideoConv3D(input_channels=1, output_channels=1)
    print('made model')

    # Train the model
    train_model(model, dataloader, device, num_epochs=5, learning_rate=0.001)
    print('trained model')

    # Save the model
    save_path = "video_conv3d.pth"
    save_model(model, save_path)
    print("saved model")

def test(save_path, data_path, entry_point):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoDataset(device, num_frames=10, entry_point=entry_point, data_path = data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    loaded_model = load_model(save_path, input_channels=1, output_channels=1, device=device)
    print('loaded model')
    test_predictions = test_model(loaded_model, dataloader, device)
    print(test_predictions)

    save_path = "real"
    np.save(save_path, test_predictions)
    print(np.shape(test_predictions))

    return test_predictions

def full_prediction(save_path, initial_frames, num_future_frames=100*60):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(save_path, input_channels=1, output_channels=1, device=device)
    print("loaded model")

    return predict_future_frames(model, initial_frames, num_future_frames, device)


if __name__ == "__main__":
    # main()
    pass
