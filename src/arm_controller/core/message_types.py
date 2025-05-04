from abc import ABC, abstractmethod
from typing import List, Any, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from pathlib import Path


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
    id: int
    frequency: int
    duration: float
    running: bool

    def __repr__(self):
        return f"SimStateMessage(id={self.id}, freq={self.frequency}, duration={self.duration}, running={self.running})"


class TimingMessage(Message):
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
        return f"TimingMessage: Current Time: {self.current_time}, dt: {self.dt}"
    
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
class PathMessage(Message):
    """A message that wraps a Path object for safe transport."""
    path: Path

    def __repr__(self):
        return f"PathMessage(path='{self.path}')"


class JointTorqueMessage(Message):
    def __init__(self, t1, t2):
        """class for joint torques"""
        self._t1 = t1
        self._t2 = t2

    @property
    def torques(self):
        return np.array([self._t1, self._t2])
    
    def __repr__(self) -> str:
        return f"TorqueMessage(t1={self._t1}, t2={self._t2})"

class TwoFloatMessage(Message, ABC):
    def __init__(self, a: Union[float, np.ndarray], b: float = None, name_a="a", name_b="b"):
        if isinstance(a, np.ndarray):
            arr = np.squeeze(a)
            if arr.shape == (2,):
                self._a = float(arr[0])
                self._b = float(arr[1])
            else:
                raise ValueError(f"Expected numpy array with shape (2,) or (1,2), got shape {arr.shape}")
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
            self._a = float(a)
            self._b = float(b)
        else:
            raise TypeError(f"{self.__class__.__name__} requires two floats or a numpy array of shape (2,)")

        self._name_a = name_a
        self._name_b = name_b

    @property
    def point(self) -> np.ndarray:
        return np.array([self._a, self._b])

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name_a}={self._a}, {self._name_b}={self._b})"



class CartesianMessage(TwoFloatMessage):
    def __init__(self, x: Union[float, np.ndarray], y: float = None):
        super().__init__(x, y, name_a="x", name_b="y")

    @property
    def x(self):
        return self._a

    @property
    def y(self):
        return self._b
    
class JointTorqueMessage(TwoFloatMessage):
    def __init__(self, t1: Union[float, np.ndarray], t2: float = None):
        super().__init__(t1, t2, name_a="t1", name_b="t2")

    @property
    def torques(self):
        return self.point
    
@dataclass(frozen=True)
class ArmDescriptionMessage(Message):
    """A message for the description of the 2dof arm"""
    l_1: float
    l_2: float
    m_1: float
    m_2: float
    g: float

    def __repr__(self):
        return f"ArmDescriptionMessage(l1={self.l_1}, l2={self.l_2}, m1={self.m_1}, m2={self.m_2}, g={self.g})"