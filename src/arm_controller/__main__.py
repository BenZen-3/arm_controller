import argparse
from .simulator import Simulator
from .arm import Arm
# from .learning import 


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

    arm = Arm(x0=0, y0=0, l1=2, l2=2, m1=1, m2=1, g=-9.8)
    sim = Simulator(800, 600, "Arm Simulator", arm)
    sim.run()

def data_collection_mode():
    print("Data Collection mode")

def train_network():
    print("Train Network mode")

if __name__ == "__main__":
    main()