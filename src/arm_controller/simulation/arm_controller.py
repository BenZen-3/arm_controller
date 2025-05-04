from abc import ABC, abstractmethod
import numpy as np

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import JointTorqueMessage, TimingMessage, ArmStateMessage, Message


class Controller(ABC):
    """Abstract base class for all controllers that output joint torques."""

    def __init__(self, message_bus: MessageBus, frequency: int = None):

        self.bus = message_bus
        self.bus.subscribe("sim/controller_update", self.controller_update)

        # set frequency
        if frequency is None:
            msg = self.bus.get_state("sim/sim_state")
            print(msg)
            self.frequency = msg.frequency

    def controller_update(self, msg: TimingMessage):
        """when the sim node sends a tick with an update to the time, the controller runs and computes torques"""

        goal_msg = self.bus.get_state("sim/goal_state")
        arm_state_msg = self.bus.get_state("arm/arm_state")
        joint_torques = self.compute_torque(arm_state_msg, goal_msg)
        self.bus.set_state("controller/controller_torque_state", JointTorqueMessage(joint_torques))

    @abstractmethod
    def compute_torque(self, arm_state_msg: ArmStateMessage, goal_msg: Message) -> np.ndarray:
        """Subclasses must implement this to return a (2,) ndarray of torques."""
        pass


class NoController(Controller):
    """A dummy controller that outputs zero torque."""

    def __init__(self, message_bus):
        super().__init__(message_bus)

    def compute_torque(self, arm_state_msg: ArmStateMessage, goal_msg: Message) -> np.ndarray:
        return np.zeros(2)