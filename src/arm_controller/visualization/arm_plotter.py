import matplotlib.pyplot as plt
import numpy as np
from arm_controller.core.message_types import ArmStateMessage

class ArmVisualizer:
    def __init__(self, l_1=1.0, l_2=1.0):
        self.l_1 = l_1
        self.l_2 = l_2

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'o-', lw=4, color='blue')
        self.ax.set_xlim(- (l_1 + l_2), l_1 + l_2)
        self.ax.set_ylim(- (l_1 + l_2), l_1 + l_2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.fig.show()
        self.fig.canvas.draw()

    def plot_state(self, state: ArmStateMessage):
        x_0, y_0 = state.x_0, state.y_0
        theta_1, theta_2 = state.theta_1, state.theta_2

        # slightly duplicate code, but tbh might be better than creating an Arm object
        x_1 = x_0 + self.l_1 * np.cos(theta_1)
        y_1 = y_0 + self.l_1 * np.sin(theta_1)
        x_2 = x_1 + self.l_2 * np.cos(theta_1 + theta_2)
        y_2 = y_1 + self.l_2 * np.sin(theta_1 + theta_2)

        self.line.set_data([x_0, x_1, x_2], [y_0, y_1, y_2])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
