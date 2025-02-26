import numpy as np

"""
musings

voxel reps should be tagged with text description - like a trasformer
each object should be its own channel of input - up to N channels of input, all tagged with text 



reason in 2D by taking slices of 3d problems and putting them in 2D


what if it became 'stateful' by labelling portions of the voxels not only with the object, but the portion of state
    its like 'inherent state'

step 1: look at image and segment into objects
step 2: look at objects and segment into components for state
step 3: predict how the state of each object will update
step 4: predict how the robot should update with the state updates

how to learn a new robot:
give it a robot and a task
tell it to get to complete a task - it guesses a voxel state
the error is how terrible the voxel state is versus the robot's ability to get to that state


if you feed a sequence of stuff to a transformer - just tag the position, who cares about feeding the useless empty space. it should rely
on the position tag, nothing more, nothing less

how can and input sequence be leveraged as 'learning a task'
    like if i add a sequence to my prompt as an example of the skill completion - that could be helpful
    then you only need one example or very few examples of skill completion

    
    
--- Another Random Idea --- 
I had a random idea, but can't seem to find anything on it (probably because I do not know what to google). 
Has anyone tried using diffusion models with only partially adding noise to an image? 
I'm thinking in terms of combining a diffusion policy with a physics simulation.

You'd be able to 'ground' the output of the network - because the physics simulation has pretty okay results in the short term - you can ground your output 
by not adding noise to the known ouput of the network - or even leave some noise based on how confident the physics result is

Then you can leave the important parts of the image fully noisy - this would allow for the model to generate from scratch
in the areas where there is a lot of noise, while leaving behind the areas that the physics engine took care of

maybe this is faster and more grounded - then your planning and manipulation and everything that your diffusion policy does can be 
physically grounded by a sim - a great benefit of an MPC - while having the excellent generative abilities and creative hallucinations of a large model



"""


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


class Arm:

    def __init__(self, x0=0, y0=0, theta1=0, theta2=0, l1=1, l2=1, m1=1, m2=1, g=-9.8):
        
        # params, not modified 
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.linkage_width = .32

        # carry the state in a separate instance for clarity between params and state vars
        self.state = ArmState(x0,y0, theta1, theta2)

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
    
    def cartesian_EE_location(self):
        """
        same thing but the last one
        """

        return self.cartesian_joint_locations()[2]
    
    def inverse_kinematics(self, x, y):
        """
        gets the joint positions that correspond to the cartesian inputs
        """
        scalar_input = np.isscalar(x) and np.isscalar(y)
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        d = np.sqrt(x**2 + y**2)
        
        b = np.arccos((self.l1**2 + self.l2**2 - x**2 - y**2) / (2 * self.l1 * self.l2))
        a = np.arccos((self.l1**2 - self.l2**2 + x**2 + y**2) / (2 * self.l1 * d))
        c = np.arctan2(y, x)
        
        theta1 = c - a
        theta2 = np.pi - b
        
        theta1, theta2 = self.smooth_angles(theta1), self.smooth_angles(theta2)
        
        return (theta1.item(), theta2.item()) if scalar_input else (theta1, theta2)
    
    def jacobian(self):
        """
        jacobian
        """
    
        # state
        theta1 = self.state.theta1
        theta2 = self.state.theta2

        # arm params
        l1 = self.l1
        l2 = self.l2

        jacobian = np.array([[-l1*np.sin(theta1) - l2*np.sin(theta1 + theta2), -l2*np.sin(theta1 + theta2)],
                         [l1*np.cos(theta1) + l2*np.cos(theta1 + theta2), l2*np.cos(theta1 + theta2)]])
        
        return jacobian

    def smooth_angles(self, angles): # TODO: THIS DOESNT BELONG HERE. THIS SCHEDULED FOR DEMOLITION
        """
        takes a list of angles and makes sure that the movements aren't insane
        """
        scalar_input = np.isscalar(angles)
        angles = np.atleast_1d(angles)
        diffs = np.diff(angles)
        
        adjustments = np.cumsum(np.where(diffs > np.pi, -2*np.pi, np.where(diffs < -np.pi, 2*np.pi, 0)))
        
        smoothed = angles[0] + np.concatenate(([0], adjustments))
        return smoothed.item() if scalar_input else smoothed
