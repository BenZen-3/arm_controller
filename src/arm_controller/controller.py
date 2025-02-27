from .arm import Arm
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class ArmGoal:
    def __init__(self, start, end, speed):
        """
        An ArmGoal is a vector defining motion from start to end at a given speed.
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.speed = speed
        
        direction = self.end - self.start
        self._direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else np.zeros_like(direction)
    
    @property
    def direction(self):
        return self._direction


class ArmPlanner:
    def __init__(self, arm, cartesian_goals, speed):
        self.arm = arm
        self.speed = speed
        self.goals = self.convert_to_theta_dot_goal(cartesian_goals)

        self.current_goal = self.goals[0] if self.goals else None
        self.proximity_threshold = 1
        self.speed_threshold = 1

    def convert_to_theta_dot_goal(self, cartesian_goals):
        """
        Converts Cartesian position goals to joint velocity goals.
        """

        theta_goals = []
        for goal in cartesian_goals:
            t1_goal, t2_goal = self.arm.inverse_kinematics(goal[0], goal[1])
            theta_goals.append((t1_goal, t2_goal))

        # theta_goals = self.arm.inverse_kinematics(cartesian_goals[0], cartesian_goals[1])

        goals_dot = []
        for i in range(len(theta_goals) - 1):
            start = theta_goals[i]
            end = theta_goals[i + 1]
            speed = self.speed if i < len(theta_goals) - 2 else 0  # Slow down at the last goal
            goals_dot.append(ArmGoal(start, end, speed))
        return goals_dot

    def goal_proximity(self, goal, arm_pos):
        return np.linalg.norm(np.array(goal.end) - np.array(arm_pos))

    def next_goal(self, arm_pos, arm_speed=0):
        """
        Computes the next goal in theta_dot space.
        """

        return np.array([1,1])

        if len(self.goals) == 1:
            return self.current_goal
        
        if self.goal_proximity(self.current_goal, arm_pos) < self.proximity_threshold: #and np.linalg.norm(arm_speed) < self.speed + self.speed_threshold:
            self.goals.pop(0)
            if self.goals:
                self.current_goal = self.goals[0]
            else:
                self.current_goal = None
        
        return self.current_goal


class ArmController:
    def __init__(self, arm, speed=3, Kp=15, Kd=3):
        self.arm = arm
        self.speed = speed
        self.Kp = Kp
        self.Kd = Kd
        self.previous_error = np.array([0,0])



    def move_to(self, goal, dt):
        """
        Move to the given ArmGoal.
        """

        current_location = self.arm.cartesian_EE_location()
        error = goal - current_location
        print(error)

        J = self.arm.jacobian()
        # J_inv = np.linalg.pinv(J)  # Use pseudo-inverse in case J is singular

        error_derivative = (error - self.previous_error) / dt
        task_space_force = self.Kp * error + self.Kd * error_derivative

        torque = J.T @ task_space_force

        self.previous_error = error

        # self.arm.state.theta1_dot *= .9
        # self.arm.state.theta2_dot *= .9

        return torque

        # self.arm.state.theta1 = goal.end[0]
        # self.arm.state.theta2 = goal.end[1]
        # self.arm.state.theta1_dot = 0
        # self.arm.state.theta2_dot = 0


        # return np.array([2,1])

        # theta_dot_goal = goal.direction * goal.speed
        # return self.effort_to_move(theta_dot_goal)

    def effort_to_move(self, goal_dot):
        """
        Moves to goal velocity (x_dot, y_dot)
        """
        J = self.arm.jacobian()
        J_inv = np.linalg.pinv(J)  # Use pseudo-inverse in case J is singular
        theta_dot_goal = J_inv @ np.array(goal_dot).reshape(2, 1)
        # theta_dot_goal = np.dot(J_inv, np.array(goal_dot))
        # print(theta_dot_goal)
        return self.velocity_PD(theta_dot_goal)

    def velocity_PD(self, theta_dot_goal):
        """
        PD loop for theta_dot tracking
        """
        current_theta_dot = np.array([[self.arm.state.theta1_dot], [self.arm.state.theta2_dot]])
        theta_dot_error = theta_dot_goal - current_theta_dot
        error_derivative = theta_dot_error - self.previous_error
        self.previous_error = theta_dot_error
        control_effort = self.Kp * theta_dot_error + self.Kd * error_derivative
        return control_effort















# class ArmSpeed(Enum): # okay yeah this is bad. this is temp
#     """
#     arm speed
#     """
#     STOP = 0
#     MEDIUM = 3
#     FASTEST = 5


    








# class Controller(ABC):

#     def __init__(self, arm: Arm, speed = ArmSpeed.MEDIUM.value):
        
#         self.arm = arm
#         self.speed = speed

#     @abstractmethod
#     def cartesian_move(self, x, y):
#         pass
    

# class CartesianPD(Controller):

#     def __init__(self, arm: Arm):
#         super().__init__(arm)

#     def joint_space_move_to(self, theta_1, theta_2):
        
#         # current_pos = 
#         pass







# class GravCompPD(CartesianPD):

#     def __init__(self, arm: Arm):
#         super().__init__(arm)


# class ComputedTorquePD(CartesianPD):

#     def __init__(self, arm: Arm):
#         super().__init__(arm)
