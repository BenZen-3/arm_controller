import numpy as np
from dataclasses import dataclass, asdict

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.subscriber import Subscriber
from arm_controller.core.publisher import Publisher
from arm_controller.core.message_types import ArmStateMessage, JointTorqueMessage

@dataclass
class ArmState:
    """Contains the internal mutable state of the 2DoF arm."""
    x_0: float = 0.0
    y_0: float = 0.0
    theta_1: float = 0.0
    theta_2: float = 0.0
    theta_1_dot: float = 0.0
    theta_2_dot: float = 0.0

class Arm:
    """2DoF arm dynamics for simulation"""

    def __init__(self, message_bus: MessageBus, x_0=0, y_0=0, theta_1=0, theta_2=0, l_1=1, l_2=1, m_1=1, m_2=1, g=9.8):
        
        self.bus = message_bus

        # arm is subscribed to joint torques. When controller gives torque, arm updates
        self.bus.subscribe("controller/joint_torques", self.state_update)

        # params, not modified 
        self.l_1 = l_1
        self.l_2 = l_2
        self.m_1 = m_1
        self.m_2 = m_2
        self.g = g

        # carry the state in a separate instance for clarity between params and state vars
        self.state = ArmState(x_0, y_0, theta_1, theta_2)

    def dynamics(self):
        """
        calculates the M, C, G matrices for dynamics of the 2DoF linkage
        """

        # state
        theta_1 = self.state.theta_1
        theta_2 = self.state.theta_2
        theta_1_dot = self.state.theta_1_dot
        theta_2_dot = self.state.theta_2_dot

        # arm params
        l_1 = self.l_1
        l_2 = self.l_2
        m_1 = self.m_1 
        m_2 = self.m_2
        g = self.g

        # Mass Matrix, M(q)
        M = np.array([
            [m_1 * l_1**2 + m_2 * (l_1**2 + 2 * l_1 * l_2 * np.cos(theta_2) + l_2**2),
            m_2 * (l_1 * l_2 * np.cos(theta_2) + l_2**2)],
            [m_2 * (l_1 * l_2 * np.cos(theta_2) + l_2**2),
            m_2 * l_2**2]
        ])

        # Coriolis and Centripetal Matrix, C(q, q_dot)
        C = np.array([
            [-m_2 * l_1 * l_2 * np.sin(theta_2) * (2 * theta_1_dot * theta_2_dot + theta_2_dot**2)],
            [m_2 * l_1 * l_2 * theta_1_dot**2 * np.sin(theta_2)]
        ])

        # Gravity Vector, G(q)
        G = np.array([
            [(m_1 + m_2) * l_1 * g * np.cos(theta_1) + m_2 * g * l_2 * np.cos(theta_1 + theta_2)],
            [m_2 * g * l_2 * np.cos(theta_1 + theta_2)]
        ])

        return M, C, G

    def dynamics_wrapper(self, theta_1, theta_2, theta_1_dot, theta_2_dot, U):
        """
        mini wrapper for the dynamics calcs so that RK4 can be run
        """

        self.state.theta_1 = theta_1
        self.state.theta_2 = theta_2
        self.state.theta_1_dot = theta_1_dot
        self.state.theta_2_dot = theta_2_dot
        
        M, C, G = self.dynamics()
        q_dd = np.linalg.solve(M, -C.flatten() - G.flatten() + U.flatten())

        return np.array([theta_1_dot, theta_2_dot, q_dd[0], q_dd[1]])

    # def state_update(self, dt=None, U=np.array([[0], [0]])): # not even sure if U shape is correct? 
    def state_update(self, msg: JointTorqueMessage):
        """
        Update the state given some dt and control vecotr U
        """

        dt = _
        U = msg.torques

        
        # RK4 Integration
        state = np.array([self.state.theta_1, self.state.theta_2, self.state.theta_1_dot, self.state.theta_2_dot])
        
        k1 = dt * self.dynamics_wrapper(*state, U)
        k2 = dt * self.dynamics_wrapper(*(state + 0.5 * k1), U)
        k3 = dt * self.dynamics_wrapper(*(state + 0.5 * k2), U)
        k4 = dt * self.dynamics_wrapper(*(state + k3), U)

        state += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6
        
        # update the state var here
        self.state.theta_1, self.state.theta_2, self.state.theta_1_dot, self.state.theta_2_dot = state

        self.bus.set_state("arm/arm_state", ArmStateMessage(**asdict(self.state))) # need to test...

    def cartesian_joint_locations(self):
        """
        where the arm joints are in cartesian
        """

        x1 = self.l_1 * np.cos(self.state.theta_1) + self.state.x_0
        y1 = self.l_1 * np.sin(self.state.theta_1) + self.state.y_0
        x2 = x1 + self.l_2 * np.cos(self.state.theta_1 + self.state.theta_2)
        y2 = y1 + self.l_2 * np.sin(self.state.theta_1 + self.state.theta_2)

        return np.array([self.state.x_0,self.state.y_0]), np.array([x1,y1]), np.array([x2,y2])
    
    def cartesian_EE_location(self):
        """
        same thing but the last one
        """

        return self.cartesian_joint_locations()[2]





    # def kinematic_move(self, x, y, smooth=True):
    #     """
    #     Purely kinematic move to x, y, maintaining solution consistency
    #     when smooth mode is enabled.
    #     """
    #     # Get the new joint angles from inverse kinematics
    #     t1_new, t2_new = self.inverse_kinematics(x, y)
        
    #     if smooth and len(self.history) > 0:

    #         # Save current state
    #         state_copy = copy.deepcopy(self.state)
    #         self.history.append(state_copy)
            
    #         # Extract angle histories
    #         t1_history = [state.theta_1 for state in self.history]
    #         t2_history = [state.theta_2 for state in self.history]
            
    #         # Add the newly calculated angles to the histories
    #         t1_history.append(t1_new)
    #         t2_history.append(t2_new)
            
    #         # Smooth both angle sequences to maintain solution consistency
    #         t1_smoothed = self.smooth_angles(t1_history)
    #         t2_smoothed = self.smooth_angles(t2_history)
            
    #         # Use the latest smoothed angles
    #         self.state.theta_1 = t1_smoothed[-1]
    #         self.state.theta_2 = t2_smoothed[-1]
    #     else:
    #         # If not smoothing or no history, just use the direct IK solution
    #         if smooth:  # First entry for history if smooth mode is on
    #             state_copy = copy.deepcopy(self.state)
    #             self.history.append(state_copy)
                
    #         self.state.theta_1 = t1_new
    #         self.state.theta_2 = t2_new
        
    #     # Limit history size to prevent memory issues
    #     max_history = 100  # Adjust as needed
    #     if len(self.history) > max_history:
    #         self.history = self.history[-max_history:]  
    
    # def inverse_kinematics(self, x, y):
    #     """
    #     Gets the joint positions that correspond to the Cartesian inputs,
    #     accounting for the base position (x_0, y_0).
    #     """

    #     scalar_input = np.isscalar(x) and np.isscalar(y)
    #     x, y = np.atleast_1d(x), np.atleast_1d(y)
        
    #     # Adjust for base position
    #     x_rel = x - self.state.x_0
    #     y_rel = y - self.state.y_0

    #     d = np.sqrt(x_rel**2 + y_rel**2)
        
    #     b = np.arccos((self.l_1**2 + self.l_2**2 - x_rel**2 - y_rel**2) / (2 * self.l_1 * self.l_2))
    #     a = np.arccos((self.l_1**2 - self.l_2**2 + x_rel**2 + y_rel**2) / (2 * self.l_1 * d))
    #     c = np.arctan2(y_rel, x_rel)
        
    #     theta_1 = c - a
    #     theta_2 = np.pi - b
        
    #     theta_1, theta_2 = self.smooth_angles(theta_1), self.smooth_angles(theta_2)
        
    #     return (theta_1.item(), theta_2.item()) if scalar_input else (theta_1, theta_2)

    # def jacobian(self):
    #     """
    #     jacobian
    #     """
    
    #     # state
    #     theta_1 = self.state.theta_1
    #     theta_2 = self.state.theta_2

    #     # arm params
    #     l_1 = self.l_1
    #     l_2 = self.l_2

    #     jacobian = np.array([[-l_1*np.sin(theta_1) - l_2*np.sin(theta_1 + theta_2), -l_2*np.sin(theta_1 + theta_2)],
    #                      [l_1*np.cos(theta_1) + l_2*np.cos(theta_1 + theta_2), l_2*np.cos(theta_1 + theta_2)]])
        
    #     return jacobian

    # # def smooth_angles(self, angles): # TODO: THIS DOESNT BELONG HERE. THIS SCHEDULED FOR DEMOLITION
    # #     """
    # #     takes a list of angles and makes sure that the movements aren't insane
    # #     """
    # #     scalar_input = np.isscalar(angles)
    # #     angles = np.atleast_1d(angles)
    # #     diffs = np.diff(angles)
        
    # #     adjustments = np.cumsum(np.where(diffs > np.pi, -2*np.pi, np.where(diffs < -np.pi, 2*np.pi, 0)))
        
    # #     smoothed = angles[0] + np.concatenate(([0], adjustments))
    # #     return smoothed.item() if scalar_input else smoothed
        

    # def smooth_angles(self, angles):
    #     """
    #     Takes a list of angles and ensures continuity by detecting and correcting large jumps
    #     that would indicate a flip between IK solutions.
        
    #     Args:
    #         angles: Array or list of angles (radians)
            
    #     Returns:
    #         Smoothed angles with consistency in solution choice
    #     """
    #     # Convert to numpy array if not already
    #     angles = np.atleast_1d(angles)
    #     scalar_input = np.isscalar(angles) or (isinstance(angles, np.ndarray) and angles.size == 1)
        
    #     if len(angles) <= 1:
    #         return angles
        
    #     # Create a copy to modify
    #     smoothed = np.copy(angles)
        
    #     # Detect large jumps (greater than π radians) and correct them
    #     # These typically indicate a flip between the two IK solutions
    #     for i in range(1, len(smoothed)):
    #         diff = smoothed[i] - smoothed[i-1]
            
    #         # If the jump is more than π radians, it's likely a solution flip
    #         # We adjust by 2π to bring it back to the same solution space
    #         if diff > np.pi:
    #             smoothed[i] -= 2 * np.pi
    #         elif diff < -np.pi:
    #             smoothed[i] += 2 * np.pi
        
    #     return smoothed.item() if scalar_input else smoothed