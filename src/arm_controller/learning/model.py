import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMLP(nn.Module):
    def __init__(self, n_gaussians=4, gmm_dim=6, num_conditioning_states=4, cond_dim=2, hidden_dim=256):
        super().__init__()

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
