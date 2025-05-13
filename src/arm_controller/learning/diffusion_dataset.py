from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path

from arm_controller.core.message_bus import MessageBus

class DiffusionDataset(Dataset):

    GENERIC_NAME = "diffusion_dataset"

    def __init__(self, bus, n_diffusion_steps, diffused_gmm_history, noise_history, noise_scale_history, state_history):
        self.bus = bus
        self.n_diffusion_steps = n_diffusion_steps
        self.num_conditioning_states = 4

        # Calculate how many initial frames to discard to align sequences
        offset = self.num_conditioning_states * self.n_diffusion_steps

        self.diffused_gmm_history = diffused_gmm_history[offset:]
        self.noise_history = noise_history[offset:]
        self.noise_scale_history = noise_scale_history[offset:]
        self.conditioning_history = self.prepare_conditioning_history(state_history)

        assert len(self.diffused_gmm_history) == len(self.noise_history) == len(self.noise_scale_history) == len(self.conditioning_history), \
            "All inputs must be the same length after alignment"

    def prepare_conditioning_history(self, state_history):
        """
        Returns conditioning history aligned with diffusion inputs.
        Shape: (total_samples, num_conditioning_states, 2)
        """
        num_conditioning_states = self.num_conditioning_states
        n_diffusion_steps = self.n_diffusion_steps

        # Extract (theta_1, theta_2) from each state
        joint_history = np.array([[s.theta_1, s.theta_2] for s in state_history], dtype=np.float32)
        total_frames = joint_history.shape[0]

        if total_frames <= num_conditioning_states:
            raise ValueError("Not enough frames for conditioning history.")

        # Build rolling window of conditioning states
        cond_sequences = np.array([
            joint_history[i - num_conditioning_states:i]
            for i in range(num_conditioning_states, total_frames)
        ])  # (frames - num_conditioning_states, num_conditioning_states, 2)

        # Repeat each conditioning sample for each diffusion step
        cond_repeated = np.repeat(cond_sequences, n_diffusion_steps, axis=0)
        return cond_repeated  # shape: (n_samples, num_conditioning_states, 2)

    def save(self):
        """
        Saves the dataset tensors to the given path using torch.save.
        """
        data = {
            "n_diffusion_steps": self.n_diffusion_steps,
            "num_conditioning_states": self.num_conditioning_states,
            "diffused_gmm_history": self.diffused_gmm_history,
            "noise_history": self.noise_history,
            "noise_scale_history": self.noise_scale_history,
            "conditioning_history": self.conditioning_history,
        }

        save_path = self.bus.get_state("common/data_directory").path
        id = self.bus.get_state("sim/sim_state").id
        file_path = save_path.joinpath(f"{id}_{self.GENERIC_NAME}.pkl")

        torch.save(data, file_path)
        print(f"Dataset saved to: {file_path}")

    @classmethod
    def load(cls, path: Path, bus: MessageBus):
        """
        Loads the dataset from the given path and returns a new instance of the dataset.
        """
        data = torch.load(path)
        dataset = cls(
            bus=bus,
            n_diffusion_steps=data["n_diffusion_steps"],
            diffused_gmm_history=data["diffused_gmm_history"],
            noise_history=data["noise_history"],
            noise_scale_history=data["noise_scale_history"],
            state_history=None  # This is bypassed because we set `conditioning_history` manually
        )
        dataset.num_conditioning_states = data["num_conditioning_states"]
        dataset.conditioning_history = data["conditioning_history"]

        print(f"Dataset loaded from: {path}")
        return dataset

    def __len__(self):
        return len(self.diffused_gmm_history)

    def __getitem__(self, idx):
        return {
            "gmm": torch.tensor(self.diffused_gmm_history[idx], dtype=torch.float32),              # (n_gaussians, 6)
            "noise_scale": torch.tensor(self.noise_scale_history[idx], dtype=torch.float32),        # ()
            "conditioning": torch.tensor(self.conditioning_history[idx], dtype=torch.float32),      # (num_conditioning_states, 2)
            "target_noise": torch.tensor(self.noise_history[idx], dtype=torch.float32)              # (n_gaussians, 6)
        }
