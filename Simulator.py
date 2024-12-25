import numpy as np
import pygame
import time

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
BLACK =     (0 ,    0,   0)

class Simulator(pygame.sprite.Sprite):
    def __init__(self, width, height, title, arm, start = np.array([1,2]), goal = np.array([2,1])):
        super().__init__()
        self.width = width
        self.height = height
        self.title = title
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

        self.arm = arm
        self.start = start
        self.goal = goal

        self.running = True
        self.fps = 100

    def run(self):

        frame_time = 1/self.fps
        
        while self.running:
            self.check_quit()

            arm.state_update(frame_time)
            self.draw_arm()

            pygame.display.flip()
            time.sleep(frame_time)
            self.screen.fill(BLACK)

        pygame.quit()

    def check_quit(self):

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

    def draw_arm(self):

        offset = np.array([self.width/2, self.height/2])

        base, j1, j2 = arm.cartesian_joint_locations()

        # print(type(base))

        pygame.draw.circle(self.screen, RED, base*50+offset, 20)
        pygame.draw.circle(self.screen, BLUE, j1*50+offset, 20)
        pygame.draw.circle(self.screen, GREEN, j2*50+offset, 20)


class Arm():

    class ArmState:

        def __init__(self, x0=0.0, y0=0.0, theta1=0.0, theta2=0.0, theta1_dot=0.0, theta2_dot=0.0):
            self.x0 = x0
            self.y0 = y0
            self.theta1 = theta1
            self.theta2 = theta2
            self.theta1_dot = theta1_dot
            self.theta2_dot = theta2_dot

    def __init__(self, x0=0, y0=0, l1=1, l2=1, m1=1, m2=1, g=-9.8):
        
        # params
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g

        # carry the state in a separate instance for clarity between params and state vars
        self.state = self.ArmState(x0,y0)

    def dynamics(self):

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

    def state_update(self, dt, U=np.array([[0], [0]])):
        def dynamics_wrapper(theta1, theta2, theta1_dot, theta2_dot):
            self.state.theta1 = theta1
            self.state.theta2 = theta2
            self.state.theta1_dot = theta1_dot
            self.state.theta2_dot = theta2_dot
            
            M, C, G = self.dynamics()
            M_inv = np.linalg.inv(M)
            q_dd = np.dot(M_inv, -C - G + U)
            
            return np.array([theta1_dot, theta2_dot, q_dd[0][0], q_dd[1][0]])
        
        # RK4 Integration
        state = np.array([self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot])
        
        k1 = dt * dynamics_wrapper(*state)
        k2 = dt * dynamics_wrapper(*(state + 0.5 * k1))
        k3 = dt * dynamics_wrapper(*(state + 0.5 * k2))
        k4 = dt * dynamics_wrapper(*(state + k3))

        state += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6
        
        self.state.theta1, self.state.theta2, self.state.theta1_dot, self.state.theta2_dot = state

    def cartesian_joint_locations(self):

        x1 = self.l1 * np.cos(self.state.theta1)
        y1 = self.l1 * np.sin(self.state.theta1)
        x2 = x1 + self.l2 * np.cos(self.state.theta1 + self.state.theta2)
        y2 = y1 + self.l2 * np.sin(self.state.theta1 + self.state.theta2)

        return np.array([self.state.x0,self.state.y0]), np.array([x1,y1]), np.array([x2,y2])


if __name__ == "__main__":

    arm = Arm()
    sim = Simulator(800, 600, "Arm Simulator", arm)

    sim.run()
    