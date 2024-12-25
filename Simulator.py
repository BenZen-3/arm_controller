import numpy as np
import pygame
import time


class CustomWindow(pygame.sprite.Sprite):
    def __init__(self, width, height, title):
        super().__init__()
        self.width = width
        self.height = height
        self.title = title
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

class Simulator:

    def __init__(self, arm, start = np.array([1,2]), goal = np.array([2,1])):

        self.arm = arm
        self.window = CustomWindow(800, 600, "Arm Simulator")

        self.start = start
        self.goal = goal

    def play(self):
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            

            # self.window.update()
            # self.window.draw()
            pygame.display.flip()

        pygame.quit()





class Arm():

    class ArmState:

        def __init__(self, x0=0, y0=0, theta1=0, theta2=0, theta1_dot=0, theta2_dot=0):
            self.x0 = x0
            self.y0 = y0
            self.theta1 = theta1
            self.theta2 = theta2
            self.theta1_dot = theta1_dot
            self.theta2_dot = theta2_dot

    def __init__(self, x0=0, y0=0, l1=1, l2=1, m1=1, m2=1, g=9.8):
        
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


        blah = [
            [m1 * l1**2 + m2 * (l1**2 + 2 * l1 * l2 * np.cos(theta2) + l2**2),
            m2 * (l1 * l2 * np.cos(theta2) + l2**2)],
            [m2 * (l1 * l2 * np.cos(theta2) + l2**2),
            m2 * l2**2]
        ]

        # print(theta2)

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

    def state_update(self, dt, U=np.array([[0],[0]])):

        M, C, G = self.dynamics()


        # print(M)
        # print(C)
        # print(G)
        # print('uahsdfasfd')
        # print((C + G))
        # print(-np.linalg.inv(M))
        # print(-np.linalg.inv(M) * (C + G))
        # print("als;jdfa;lsfdj")
        # print(np.dot(-np.linalg.inv(M), (C+G)))



        M_inv = np.linalg.inv(M)
        f_X = np.dot(-M_inv, (C + G)) # f(X)
        B = M_inv

        # theta updates
        self.state.theta1 += self.state.theta1_dot * dt
        self.state.theta2 += self.state.theta2_dot * dt

        # theta dot updates
        theta_dot = f_X + np.dot(B,U)
        self.state.theta1_dot = theta_dot[0][0] * dt
        self.state.theta2_dot = theta_dot[1][0] * dt


        # print(f"{f_X=}")
        # print(f"{theta_dot=}")
        # print(f"{np.dot(B,U)=}")
        # print("jfagsfkuhlhlkasdfkhl")
        # print(theta_dot)
        # print(theta_dot[0] * dt)

        # print(f_X)

    def cartesian_joint_locations(self):

        x1 = self.l1 * np.cos(self.state.theta1)
        y1 = self.l1 * np.sin(self.state.theta1)
        x2 = x1 + self.l2 * np.cos(self.state.theta1 + self.state.theta2)
        y2 = y1 + self.l2 * np.sin(self.state.theta1 + self.state.theta2)

        return (self.state.x0,self.state.y0), (x1,y1), (x2,y2)


if __name__ == "__main__":


    arm = Arm()

    while True:

        arm.state_update(.05)
        base, j1, j2 = arm.cartesian_joint_locations()
        print(f"Joints: {j1[0]}, {j1[1]}")
        time.sleep(.05)

    # pygame.init()

    # arm = Arm()
    # sim = Simulator(arm)