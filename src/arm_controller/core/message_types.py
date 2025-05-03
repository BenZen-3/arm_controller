from abc import ABC, abstractmethod
from typing import List, Any
from pathlib import Path
from dataclasses import dataclass
import numpy as np

class Message(ABC):
    """Base class for all messages."""
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class ListMessage(list, Message):
    def __init__(self, items: List[Any]):
        """Class for any List-type messages"""
        if not isinstance(items, list):
            raise TypeError("ListMessage expects a list of items.")
        # Initialize the parent list class to use all its functionality instead
        super().__init__(items)

    def __repr__(self):
        return super().__repr__()

class NumberMessage(float, Message):
    def __new__(cls, number: float):
        """Class for numeric messages (int or float)."""
        if not isinstance(number, (int, float)):
            raise TypeError("NumberMessage expects an int or float.")
        # Use `float.__new__()` to properly initialize the float part
        return float.__new__(cls, number)

    def __repr__(self):
        return super().__repr__()
    
class FilePath(Path, Message): # SHOULD BE TESTED. THIS SEEMS WONKY
    """Class for file path-based messages."""
    
    def __new__(cls, path: str) -> 'FilePath':
        """Initialize with a valid file path."""
        if not isinstance(path, str):
            raise TypeError("FilePath expects a string representing the file path.")
        
        # Initialize the parent class (Path)
        return super().__new__(cls, path)

    def __repr__(self):
        """Return a string representation of the file path."""
        return f"FilePath({str(self)})"
    
class StringMessage(str, Message):
    def __new__(cls, text: str):
        """Ensure the text is a valid string message."""
        if not isinstance(text, str):
            raise TypeError("StringMessage expects a string.")
        
        # Return a new instance of the subclass (which will be a str object)
        return super().__new__(cls, text)

    def __repr__(self):
        return super().__repr__()

class DictMessage(dict, Message):
    def __new__(cls, data: dict):
        """Ensure the data is a valid dictionary message."""
        if not isinstance(data, dict):
            raise TypeError("DictMessage expects a dictionary.")
        
        # Create and return a new instance of the subclass (which will be a dict object)
        return super().__new__(cls, data)

def __repr__(self):
        return super().__repr__()

@dataclass(frozen=False)
class SimStateMessage(Message):
    frequency: int
    duration: float
    running: bool

    def __repr__(self):
        return f"SimStateMessage(freq={self.freq}, duration={self.duration}, running={self.running})"


class JointTorqueMessage(Message):
    def __init__(self, t1, t2):
        """class for joint torques"""
        self._t1 = t1
        self._t2 = t2

    @property
    def torques(self):
        return np.array([self._t1, self._t2])

class SimTimerMessage(Message):
    def __init__(self, current_time: float, dt: float):
        """class for joint torques"""
        self._current_time = current_time
        self._dt = dt

    @property
    def current_time(self):
        return self._current_time

    @property
    def dt(self):
        return self._dt

    def __repr__(self):
        return f"SimTimerMessage: Current Time: {self.current_time}, dt: {self.dt}"
    
# class JointStateMessage(Message):
#     def __init__(self, theta_1: float, theta_2: float):
#         """class for joint states message"""
#         self._theta_1 = theta_1
#         self._theta_2 = theta_2

#     @property
#     def theta_1(self):
#         return self._theta_1

#     @property
#     def theta_2(self):
#         return self._theta_2

#     def __repr__(self):
#         return f"JointStateMessage: theta_1: {self.theta_1}, theta_2: {self.theta_2}"

@dataclass(frozen=True)
class ArmStateMessage(Message):
    """
    immutable message type for Arm State. 
    SHOULD MATCH THE ArmState CLASS (not enforced explicitly)
    """
    x_0: float
    y_0: float
    theta_1: float
    theta_2: float
    theta_1_dot: float
    theta_2_dot: float

def __repr__(self):
    return (f"ArmStateMessage(x_0={self.x_0}, y_0={self.y_0}, "
            f"theta_1={self.theta_1}, theta_2={self.theta_2}, "
            f"theta_1_dot={self.theta_1_dot}, theta_2_dot={self.theta_2_dot})")


@dataclass(frozen=True)
class CartesianMessage(Message):
    x: float
    y: float

    def __repr__(self):
        return f"CartesianMessage(x={self.x}, y={self.y})"
