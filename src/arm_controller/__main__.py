import argparse
from .simulator import BatchProcessor
from .learning import  train, test
import time
from pathlib import Path


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
        choices=['simulator', 'collect_data', 'train_network', 'test_model'], 
        default='test_model',
        help='Choose which module to run'
    )
    args = parser.parse_args()
    
    if args.mode == 'simulator':
        simulator_mode()

    elif args.mode == 'collect_data':
        data_collection_mode()
    elif args.mode == 'train_network':
        train_network()
    elif args.mode == 'test_model':
        test_model()

def simulator_mode():
    print("Simulator mode")

    # # units of meters
    # width = 4.1
    # height = 4.1

    # arm = Arm(x0=width/2, y0=height/2, l1=1, l2=1, m1=1, m2=1, g=-9.8)
    # sim = Simulator(width, height, arm, voxel_size=.05)
    # print('starting')
    # start = time.time()
    # recording = sim.run(100)
    # # profile_func(sim.run, 100)
    # print("done recording")
    # print(time.time() - start)

    # player = SimulationPlayer(800, 800)
    # player.play(recording)

    start = time.time()
    BatchSim = BatchProcessor(40, 20)
    results = BatchSim.batch_process()
    # print(results)
    print(time.time() - start)

def data_collection_mode():
    print("Data Collection mode")
    start = time.time()
    entry_point = Path.cwd().parent # eventually clean up this weird path thing
    
    BatchSim = BatchProcessor(10, 10)
    BatchSim.batch_process(entry_point) 

    print(f"Total batch sim time: {round(time.time() - start, 2)} seconds")

def train_network():
    print("Train Network mode")
    entry_point = Path.cwd().parent # eventually clean up this weird path thing

    train(entry_point)

def test_model():
    print("testing model")

    save_path = "video_conv3d.pth"
    data = 
    test(save_path, data)


if __name__ == "__main__":
    main()