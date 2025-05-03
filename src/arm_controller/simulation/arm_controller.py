from abc import ABC, abstractmethod
import numpy as np

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import JointTorqueMessage


class Controller(ABC):
    """Abstract base class for all controllers that output joint torques."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.bus.subscribe("sim/sim_timer", self.run_controller)
        self.joint_torque_publisher = Publisher(self.bus, "controller/joint_torques")

    def run_controller(self, _msg=None):
        """when the sim node sends a tick with an update to the time, the controller runs and computes torques"""
        joint_torques = self.compute_torque()
        self.joint_torque_publisher.publish(JointTorqueMessage(joint_torques))

    @abstractmethod
    def compute_torque(self) -> np.ndarray:
        """Subclasses must implement this to return a (2,) ndarray of torques."""
        pass


class NoController(Controller):
    """A dummy controller that outputs zero torque."""

    def __init__(self, message_bus):
        super().__init__(message_bus)

    def compute_torque(self) -> np.ndarray:
        return np.zeros(2)