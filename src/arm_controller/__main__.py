import argparse
from .simulator import Simulator, SimulationPlayer, VoxelState, Recording
from .arm import Arm
# from .learning import 
import numpy as np
import time

import cProfile
import pstats

def profile_func(func, *args):
    with cProfile.Profile() as pr:
        func(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()

def main():
    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['simulator', 'collect_data', 'train_network'], 
        default='simulator',
        help='Choose which module to run'
    )
    args = parser.parse_args()
    
    if args.mode == 'simulator':
        simulator_mode()

    elif args.mode == 'collect_data':
        data_collection_mode
    elif args.mode == 'train_network':
        train_network()

def simulator_mode():
    print("Simulator mode")

    # units of meters
    width = 4.1
    height = 4.1

    arm = Arm(x0=width/2, y0=height/2, l1=1, l2=1, m1=1, m2=1, g=-9.8)
    sim = Simulator(width, height, arm, voxel_size=.05)
    print('starting')
    start = time.time()
    # recording = sim.run(100)
    profile_func(sim.run, 100)
    print("done recording")
    print(time.time() - start)

    # player = SimulationPlayer(800, 800)
    # player.play(recording)

def data_collection_mode():
    print("Data Collection mode")

def train_network():
    print("Train Network mode")

if __name__ == "__main__":
    main()