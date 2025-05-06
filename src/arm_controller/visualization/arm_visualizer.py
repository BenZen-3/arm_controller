import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List

from arm_controller.core.message_types import ArmStateMessage

# TODO: add type hinting to this vibe coded POS
# TODO: add reasonable docstrings 

class ArmVisualizer:

    DEFAULT_PLAYBACK_HZ = 30

    def __init__(self, arm_states: List[ArmStateMessage], playback_speed: float=1.0, l_1: float=1.0, l_2: float=1.0, data_collection_hz: int=100):
        """
        arm_states: list of ArmStateMessage
        data_collection_hz: original frequency of data collection
        playback_rate_hz: desired visualization playback frequency
        playback_speed: multiplier for playback speed (1.0 = normal, 2.0 = 2x speed, etc.)
        """
        self.playback_speed = playback_speed
        self.l_1 = l_1
        self.l_2 = l_2
        self.data_collection_hz = data_collection_hz
        self.interval_ms = (1000 / self.DEFAULT_PLAYBACK_HZ) / self.playback_speed  # ms/frame adjusted for playback speed
        self.states = self._downsample(arm_states, data_collection_hz, self.DEFAULT_PLAYBACK_HZ)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots()
        plt.title(f"Arm: {r'$l_{1}$'}={l_1}, {r'$l_{2}$'}={l_2}. Simulated at {data_collection_hz}hz. Speed: {playback_speed}")
        self.line, = self.ax.plot([], [], 'o-', lw=4, color='blue')
        self.ax.set_xlim(- (l_1 + l_2) + 0.1, (l_1 + l_2) + 0.1)
        self.ax.set_ylim(- (l_1 + l_2) + 0.1, (l_1 + l_2) + 0.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

    def _downsample(self, states, sim_rate, target_rate):
        """
        Downsamples the states list to match the target playback rate, using interpolation of frame indices.
        """
        num_input_frames = len(states)
        total_time = num_input_frames / sim_rate
        num_frames = int(total_time * target_rate)
        sample_times = np.linspace(0, num_input_frames - 1, num=num_frames).astype(int)
        return [states[i] for i in sample_times]

    def _update(self, frame):
        """private update function for the animation"""
        state = self.states[frame]
        x_0, y_0 = state.x_0, state.y_0
        theta_1, theta_2 = state.theta_1, state.theta_2

        # slightly duplicate code, but tbh might be better than creating an Arm object
        x_1 = x_0 + self.l_1 * np.cos(theta_1)
        y_1 = y_0 + self.l_1 * np.sin(theta_1)
        x_2 = x_1 + self.l_2 * np.cos(theta_1 + theta_2)
        y_2 = y_1 + self.l_2 * np.sin(theta_1 + theta_2)

        self.line.set_data([x_0, x_1, x_2], [y_0, y_1, y_2])
        return self.line,

    def play(self):
        """function animation"""
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.states),
            interval=self.interval_ms,
            blit=True,
            repeat=False
        )
        plt.show()
