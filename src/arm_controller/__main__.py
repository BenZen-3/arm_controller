import argparse
from .simulator import Simulator, SimulationPlayer, VoxelState, Recording
from .arm import Arm
# from .learning import 
import numpy as np


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
    width = 4.2
    height = 4.2

    arm = Arm(x0=width/2, y0=height/2, l1=1, l2=1, m1=1, m2=1, g=-9.8)
    sim = Simulator(width, height, arm, voxel_size=.2)
    recording = sim.run(10)
    print("done recording")

    # print(recording.frame_sequence)
    # print(np.flip(recording.frame_sequence,0))
    # print(np.flip(recording.frame_sequence,1))

    # print(np.shape(recording.frame_sequence))
    
    player = SimulationPlayer(800, 800)
    player.play(recording)

    # Recording.frame_printer(recording.frame_sequence[30])

    # print(len(recording.frame_sequence))

    # print(np.shape(recording.frame_sequence))

    # for row in recording.frame_sequence[30]:

    #     row_output = ""
    #     for voxel in row:
    #         if voxel.state != VoxelState.NO_FILL:
    #             row_output += "O"
    #         else:
    #             row_output += " "

    #     print(row_output)

    # for frame in recording.frame_sequence:
    #     Recording.frame_printer(frame)


    # player = SimulationPlayer(sim.width, sim.height)
    # player.play(recording)


def data_collection_mode():
    print("Data Collection mode")

def train_network():
    print("Train Network mode")

if __name__ == "__main__":
    main()