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
        self.block_size = 10 # set the size of the grid
        self.pixel_per_meter = 100

    def run(self):

        frame_time = 1/self.fps
        
        while self.running:
            self.check_quit()

            arm.state_update(frame_time)
            self.draw_all()
            time.sleep(frame_time)

        pygame.quit()

    def check_quit(self):

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

    def draw_all(self):

        self.draw_grid()
        self.draw_arm()
        self.check_grid_occupancy()

        # pygame stuff
        pygame.display.flip()
        self.screen.fill(BLACK)

    def draw_arm(self):

        offset = np.array([self.width/2, self.height/2])

        base, j1, j2 = arm.cartesian_joint_locations()
        arm_pix_width = round(arm.linkage_width*self.pixel_per_meter)

        base_pix = np.array(base*self.pixel_per_meter + offset).astype(int)
        j1_pix   = np.array(j1*self.pixel_per_meter + offset).astype(int)
        j2_pix   = np.array(j2*self.pixel_per_meter + offset).astype(int)

        # arm links
        radius = round(arm_pix_width/2)
        pygame.draw.circle(self.screen, RED, base_pix, radius)
        pygame.draw.line(self.screen, RED, base_pix, j1_pix, arm_pix_width)
        pygame.draw.circle(self.screen, BLUE, j1_pix, radius)
        pygame.draw.line(self.screen, BLUE, j1_pix, j2_pix, arm_pix_width)
        pygame.draw.circle(self.screen, GREEN, j2_pix, radius)

    def draw_grid(self):

        for x in range(0, self.width, self.block_size):
            for y in range(0, self.height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    # def check_grid_occupancy(self):

    #     filled_cells = []

    #     for x in range(0, self.width, self.block_size):
    #         for y in range(0, self.height, self.block_size):
                
    #             if (True):
    #                 filled_cells.append(1)

        # return filled_cells

    def check_grid_occupancy(self):

        filled_cells = []
        offset = np.array([self.width/2, self.height/2])

        # Get arm joint locations in pixel space
        base, j1, j2 = self.arm.cartesian_joint_locations()
        base_pix = np.array(base * self.pixel_per_meter + offset).astype(int)
        j1_pix = np.array(j1 * self.pixel_per_meter + offset).astype(int)
        j2_pix = np.array(j2 * self.pixel_per_meter + offset).astype(int)

        # Define line segments
        arm_segments = [(base_pix, j1_pix), (j1_pix, j2_pix)]
        joint_positions = [base_pix, j1_pix, j2_pix]

        # Check each grid cell
        for x in range(0, self.width, self.block_size):
            for y in range(0, self.height, self.block_size):
                cell_rect = pygame.Rect(x, y, self.block_size, self.block_size)
                
                # Check if any joint is inside the cell
                for joint in joint_positions:
                    if cell_rect.collidepoint(joint[0], joint[1]):
                        filled_cells.append((x, y))
                        pygame.draw.rect(self.screen, GREEN, cell_rect)
                        break  # No need to check further if a joint is inside

                # Check if any segment intersects the cell
                for segment in arm_segments:
                    if self.line_intersects_rect(segment[0], segment[1], cell_rect):
                        filled_cells.append((x, y))
                        pygame.draw.rect(self.screen, GREEN, cell_rect)
                        break

        return filled_cells

    def line_intersects_rect(self, p1, p2, rect):
        """Check if a line segment (p1 to p2) intersects with a rectangle."""
        rect_lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),
            ((rect.right, rect.top), (rect.right, rect.bottom)),
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),
            ((rect.left, rect.bottom), (rect.left, rect.top))
        ]
        
        for r1, r2 in rect_lines:
            if self.line_intersects_line(p1, p2, r1, r2):
                return True
        return False

    def line_intersects_line(self, p1, p2, q1, q2):
        """Check if two line segments (p1 to p2 and q1 to q2) intersect."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))







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
        self.linkage_width = .32

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

    arm = Arm(x0=0, y0=0, l1=2, l2=1, m1=1, m2=1, g=-9.8)
    sim = Simulator(800, 600, "Arm Simulator", arm)

    sim.run()
    