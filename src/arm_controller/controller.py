from .arm import Arm
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import json


class Goal:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
    
    def __repr__(self):
        return f"Goal(x={self.x}, y={self.y}, speed={self.speed})"
    
    @classmethod
    def distance(cls, goal_1, goal_2):
        """
        distance between two goals
        """

        pos_1 = np.array([goal_1.x, goal_1.y])
        pos_2 = np.array([goal_2.x, goal_2.y])

        return np.linalg.norm(pos_1 - pos_2)


class ArmTrajectory:

    def __init__(self, json_sequence):
        """
        arm_start and json_sequence are all cartesian. 
        ASSUMES THAT THE FIRST GOAL IS THE CURRENT ARM POSITION
        """
        self.goal_sequence = self.from_json(json_sequence)
        self.current_goal_index = 0

    def from_json(self, json_sequence):
        """
        Parse a JSON string or dictionary into a sequence of Goal objects.
        """
        # If json_sequence is a string, parse it into a dictionary
        if isinstance(json_sequence, str):
            json_data = json.loads(json_sequence)
        else:
            # If it's already a dictionary, use it directly
            json_data = json_sequence
        
        goal_sequence = []
        
        # Access the coordinate_list from the json data
        if "coordinate_list" in json_data:
            coordinates = json_data["coordinate_list"]
            
            # Iterate through coordinates and create Goal objects
            for point in coordinates:
                x = point.get("x", 0)
                y = point.get("y", 0)
                speed = point.get("speed", 3)  # Default speed if not specified
                
                # Create a new Goal object and add it to the sequence
                goal = Goal(x, y, speed)
                goal_sequence.append(goal)
        
        return goal_sequence

    def next(self):
        """
        Get next goal. None if there are no more goals
        """
        if self.current_goal_index < len(self.goal_sequence):
            current_goal = self.goal_sequence[self.current_goal_index]
            self.current_goal_index += 1
            return current_goal
        
        return None


class KinematicController:
    def __init__(self, arm, trajectory):
        """
        Initialize the kinematic controller with an arm and trajectory.
        
        Args:
            arm: The arm object with inverse_kinematics method
            trajectory: An ArmTrajectory object containing a sequence of Goals
        """
        self.arm = arm
        self.trajectory = trajectory
 
        # set state
        self.previous_goal = self.trajectory.next()
        self.current_goal = self.trajectory.next()
        self.progress = 0.0  # Progress towards current goal (0 to 1)
        self.current_time = 0.0
        
        # If there's no goal, we're done
        if self.previous_goal is None or self.current_goal is None:
            self.is_complete = True
        else:
            # Move to previous goal because it should start there
            self.arm.kinematic_move(self.previous_goal.x, self.previous_goal.y, smooth=False)
            self.is_complete = False
            self.set_path_time()
            
    def set_path_time(self):
            """
            sets the total_time var
            """
            
            # Calculate initial path parameters using Goal.distance
            path_length = Goal.distance(self.previous_goal, self.current_goal)

            if self.current_goal.speed > 0:
                self.total_time = path_length / self.current_goal.speed
            else:
                self.total_time = 1 # If the speed is zero, or somehow magically less than 0... then time is one second (should be a pause)

    def arm_update(self, dt):
        """
        Update the arm's position based on the current trajectory and time step.
        
        Args:
            dt: Time step in seconds
        
        Returns:
            bool: True if trajectory is completed, False otherwise
        """
        if self.is_complete:
            return True
            
        # Update time
        self.current_time += dt
        
        # Calculate progress along current segment
        if self.total_time > 0:
            self.progress = min(self.current_time / self.total_time, 1.0)
        else:
            self.progress = 1.0  # Instant movement for zero time
            
        # Interpolate position
        current_position = self._interpolate_position(self.progress)
        
        # Apply inverse kinematics to update arm angles
        x, y = current_position[0], current_position[1]
        # print(f"{x=}, {y=}")
        self.arm.kinematic_move(x, y)
        
        # Check if we've reached the current goal
        if self.progress >= 1.0:
            # Move to next goal
            self._advance_to_next_goal()
            
        return self.is_complete
    
    def _advance_to_next_goal(self):
        """
        Advance to the next goal in the trajectory.
        """
        # Store current goal as previous goal
        self.previous_goal = self.current_goal
        
        # Get next goal
        self.current_goal = self.trajectory.next()
        
        # Reset timing
        self.current_time = 0.0
        self.progress = 0.0
        
        # If there's no next goal, we're done
        if self.current_goal is None:
            self.is_complete = True
            return
            
        self.set_path_time()
    
    def _interpolate_position(self, progress):
        """
        Linearly interpolate between previous goal and current goal.
        
        Args:
            progress: Float between 0 and 1 representing progress along path
            
        Returns:
            numpy.ndarray: [x, y] interpolated position
        """
        prev_pos = np.array([self.previous_goal.x, self.previous_goal.y])
        curr_pos = np.array([self.current_goal.x, self.current_goal.y])
        
        return prev_pos + progress * (curr_pos - prev_pos)
    



    
# class ArmGoal:
#     def __init__(self, start, end, speed):
#         """
#         An ArmGoal is a vector defining motion from start to end at a given speed.
#         """
#         self.start = np.array(start)
#         self.end = np.array(end)
#         self.speed = speed
        
#         direction = self.end - self.start
#         self._direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else np.zeros_like(direction)
    
#     @property
#     def direction(self):
#         return self._direction


# class ArmPlanner:
#     def __init__(self, arm, cartesian_goals, speed):
#         self.arm = arm
#         self.speed = speed
#         self.goals = self.convert_to_theta_dot_goal(cartesian_goals)

#         self.current_goal = self.goals[0] if self.goals else None
#         self.proximity_threshold = 1
#         self.speed_threshold = 1

#     def convert_to_theta_dot_goal(self, cartesian_goals):
#         """
#         Converts Cartesian position goals to joint velocity goals.
#         """

#         theta_goals = []
#         for goal in cartesian_goals:
#             t1_goal, t2_goal = self.arm.inverse_kinematics(goal[0], goal[1])
#             theta_goals.append((t1_goal, t2_goal))

#         # theta_goals = self.arm.inverse_kinematics(cartesian_goals[0], cartesian_goals[1])

#         goals_dot = []
#         for i in range(len(theta_goals) - 1):
#             start = theta_goals[i]
#             end = theta_goals[i + 1]
#             speed = self.speed if i < len(theta_goals) - 2 else 0  # Slow down at the last goal
#             goals_dot.append(ArmGoal(start, end, speed))
#         return goals_dot

#     def goal_proximity(self, goal, arm_pos):
#         return np.linalg.norm(np.array(goal.end) - np.array(arm_pos))

#     def next_goal(self, arm_pos, arm_speed=0):
#         """
#         Computes the next goal in theta_dot space.
#         """

#         return np.array([1,1]) + np.array([2.1, 2.1])

#         if len(self.goals) == 1:
#             return self.current_goal
        
#         if self.goal_proximity(self.current_goal, arm_pos) < self.proximity_threshold: #and np.linalg.norm(arm_speed) < self.speed + self.speed_threshold:
#             self.goals.pop(0)
#             if self.goals:
#                 self.current_goal = self.goals[0]
#             else:
#                 self.current_goal = None
        
#         return self.current_goal


class DynamicArmController:
    """
    cartesian PD or joint space PD computed torque controller
    """
    def __init__(self, arm, speed=3, Kp=1, Kd=2):
        self.arm = arm
        self.speed = speed
        self.Kp = Kp
        self.Kd = Kd
        self.previous_error = np.array([0,0])

    def move_to(self, goal, dt):
        """
        Move to the given ArmGoal.
        """

        return self.cartesian_PD(goal, dt)

    def cartesian_PD(self, goal, dt):

        # Inputs
        current_location = self.arm.cartesian_EE_location()
        error = goal - current_location

        # Cartesian PD
        error_derivative = (error - self.previous_error) / dt
        self.previous_error = error
        task_space_accel = self.Kp * error + self.Kd * error_derivative

        # Desired joint space accels
        J = self.arm.jacobian()
        lambda_damping = 0.01  # Small damping factor
        J_inv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(J.shape[0]))
        joint_space_accel = J_inv @ task_space_accel

        # Computed Torque Control
        M, C, G = self.arm.dynamics()
        control_T = M @ joint_space_accel
        return C.flatten() + G.flatten() + control_T


    def joint_space_PD(self, goal, dt):

        # joint space PD 
        current_thetas = np.array([self.arm.state.theta1, self.arm.state.theta2])
        goal_thetas = self.arm.inverse_kinematics(goal[0], goal[1])

        error = np.array(goal_thetas) - np.array(current_thetas)
        error_derivative = (error - self.previous_error) / dt
        self.previous_error = error

        M, C, G = self.arm.dynamics()        
        joint_accels = self.Kp * error + self.Kd * error_derivative
        control_T = M @ joint_accels

        return C.flatten() + G.flatten() + control_T

