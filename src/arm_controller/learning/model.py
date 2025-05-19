import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

from arm_controller.core.message_bus import MessageBus
from arm_controller.learning.diffusion_dataset import DiffusionDataset

class DiffusionMLP(nn.Module):
    def __init__(self, bus: MessageBus, n_gaussians=4, gmm_dim=6, num_conditioning_states=4, cond_dim=2, hidden_dim=256):
        super().__init__()

        self.bus = bus
        self.gmm_input_dim = n_gaussians * gmm_dim
        self.cond_input_dim = num_conditioning_states * cond_dim
        self.total_input_dim = self.gmm_input_dim + self.cond_input_dim + 1  # +1 for noise scale

        # Encoder MLP
        self.model = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.gmm_input_dim),  # Predict full GMM noise
        )

    def forward(self, diffused_gmm, conditioning, noise_scale):
        """
        Inputs:
            diffused_gmm: shape (B, n_gaussians, 6)
            conditioning: shape (B, num_conditioning_states, 2)
            noise_scale: shape (B, 1)
        Output:
            predicted_noise: shape (B, n_gaussians, 6)
        """
        B = diffused_gmm.shape[0]

        # Flatten inputs
        gmm_flat = diffused_gmm.view(B, -1)             # (B, n_gaussians * 6)
        cond_flat = conditioning.view(B, -1)            # (B, num_conditioning_states * 2)
        noise_scale = noise_scale.view(B, 1)            # (B, 1)

        # Concatenate all inputs
        x = torch.cat([gmm_flat, cond_flat, noise_scale], dim=-1)  # (B, total_input_dim)

        # Predict noise
        predicted_noise = self.model(x)  # (B, n_gaussians * 6)

        # Reshape to original GMM shape
        return predicted_noise.view(B, -1, 6)  # (B, n_gaussians, 6)

    def get_full_dataset(self):
        all_datasets = []

        # Load all datasets
        data_path = self.bus.get_state("common/data_directory").path

        for file_path in data_path.glob("*.pkl"):
            dataset = DiffusionDataset.load(file_path, self.bus)
            all_datasets.append(dataset)

        # Concatenate all datasets
        return torch.utils.data.ConcatDataset(all_datasets)

    def train_model(self, num_epochs=10, batch_size=64, learning_rate=1e-4):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.to(device)

        full_dataset = self.get_full_dataset()

        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            self.train()

            running_loss = 0.0
            old_percent = -1

            for batch_num, batch in enumerate(dataloader):
                gmm, scale, cond, target_noise = batch['gmm'], batch['noise_scale'], batch['conditioning'], batch['target_noise']

                gmm = gmm.to(device)
                cond = cond.to(device)
                scale = scale.to(device)
                target_noise = target_noise.to(device)

                optimizer.zero_grad()
                pred_noise = self.forward(gmm, cond, scale)
                loss = loss_fn(pred_noise, target_noise)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                running_loss += loss.item()

                # garbo past here
                percent = round(batch_num / len(dataloader) * 100)
                if percent != old_percent:
                    print(f"Epoch [{epoch+1}/{num_epochs}] Progress: {percent}%, Loss: {running_loss/(batch_num+1)}")
                    old_percent = percent

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    def save_model(self, path: Path):
        """
        Save the entire model (architecture + weights).
        """
        torch.save(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: Path, map_location=None):
        """
        Load a saved DiffusionMLP model from disk.
        Note: Use only with trusted model files, as torch.load uses pickle.
        """
        model = torch.load(path, map_location=map_location or torch.device("cpu"))
        print(f"Model loaded from {path}")
        return model
