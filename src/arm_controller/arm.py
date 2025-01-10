import numpy as np

class Arm():

    class ArmState:
        """
        contains all the stateful things that the arm has
        """

        def __init__(self, x0=0.0, y0=0.0, theta1=0.0, theta2=0.0, theta1_dot=0.0, theta2_dot=0.0):
            self.x0 = x0
            self.y0 = y0
            self.theta1 = theta1
            self.theta2 = theta2
            self.theta1_dot = theta1_dot
            self.theta2_dot = theta2_dot

    def __init__(self, x0=0, y0=0, l1=1, l2=1, m1=1, m2=1, g=-9.8):
        
        # params, not modified 
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.linkage_width = .32

        # carry the state in a separate instance for clarity between params and state vars
        self.state = self.ArmState(x0,y0)

    def dynamics(self):
        """
        calculates the M, C, G matrices for dynamics of the 2DoF linkage
        """

        # state
        theta1 = self.state.theta1
        theta2 = self.state.theta2
        theta1_dot = self.state.theta1_dot
        theta2_dot = self.state.theta2_dot

        # arm params
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1 
        m2 = self.m2
        g = self.g

        # Mass Matrix, M(q)
        M = np.array([
            [m1 * l1**2 + m2 * (l1**2 + 2 * l1 * l2 * np.cos(theta2) + l2**2),
            m2 * (l1 * l2 * np.cos(theta2) + l2**2)],
            [m2 * (l1 * l2 * np.cos(theta2) + l2**2),
            m2 * l2**2]
        ])

        # Coriolis and Centripetal Matrix, C(q, q_dot)
        C = np.array([
            [-m2 * l1 * l2 * np.sin(theta2) * (2 * theta1_dot * theta2_dot + theta2_dot**2)],
            [m2 * l1 * l2 * theta1_dot**2 * np.sin(theta2)]
        ])

        # Gravity Vector, G(q)
        G = np.array([
            [(m1 + m2) * l1 * g * np.cos(theta1) + m2 * g * l2 * np.cos(theta1 + theta2)],
            [m2 * g * l2 * np.cos(theta1 + theta2)]
        ])

        return M, C, G

    def dynamics_wrapper(self, theta1, theta2, theta1_dot, theta2_dot, U):
        """
        mini wrapper for the dynamics calcs so that RK4 can be run
        """
        # note this flatten stuff was chatgpt and not verified. below is what I know is right
        # M_inv = np.linalg.inv(M)
        # q_dd = np.dot(M_inv, -C - G + U)
        # return np.array([theta1_dot, theta2_dot, q_dd[0][0], q_dd[1][0]])

        self.state.theta1 = theta1
        self.state.theta2 = theta2
        self.state.theta1_dot = theta1_dot
        self.state.theta2_dot = theta2_dot
        
        M, C, G = self.dynamics()
        q_dd = np.linalg.solve(M, -C.flatten() - G.flatten() + U.flatten())

        return np.array([theta1_dot, theta2_dot, q_dd[0], q_dd[1]])

    def state_update(self, dt, U=np.array([[0], [0]])): # not even sure if that shape is correct? 
        """
        Update the state given some dt and control vecotr U
        """
        
        # RK4 Integration
        state = np.array([self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot])
        
        k1 = dt * self.dynamics_wrapper(*state, U)
        k2 = dt * self.dynamics_wrapper(*(state + 0.5 * k1), U)
        k3 = dt * self.dynamics_wrapper(*(state + 0.5 * k2), U)
        k4 = dt * self.dynamics_wrapper(*(state + k3), U)

        state += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6
        
        # update the state var here
        self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot = state

    def cartesian_joint_locations(self):
        """
        where the arm joints are in cartesian
        """

        x1 = self.l1 * np.cos(self.state.theta1) + self.state.x0
        y1 = self.l1 * np.sin(self.state.theta1) + self.state.y0
        x2 = x1 + self.l2 * np.cos(self.state.theta1 + self.state.theta2)
        y2 = y1 + self.l2 * np.sin(self.state.theta1 + self.state.theta2)

        return np.array([self.state.x0,self.state.y0]), np.array([x1,y1]), np.array([x2,y2])
    
