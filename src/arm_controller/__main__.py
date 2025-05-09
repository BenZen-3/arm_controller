import argparse
import time
from pathlib import Path

from arm_controller.core.message_bus import MessageBus
from arm_controller.core.message_types import PathMessage
from arm_controller.simulation.sim_manager import SimManager
from arm_controller.data_synthesis.sim_observer import Observer

"""
todo:
    Files that are ChatGPT (or older) and require some attention:
        gmm_visualizer.py
        arm_visualizer.py
        gmm_estimator.py
        probability.py
"""


def main():
    """main program start"""

    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['generator', 'trainer', 'inference', 'visualize', "view_recording", "testing"], 
        default='generator',
        help='Choose which module to run. Generate data, train on data, or predict a sequence from a random starting state'
    )

    args = parser.parse_args()
    start = time.time()
    bus = MessageBus()
    set_public_states(bus)

    if args.mode == 'generator':
        generate_data(bus)
    elif args.mode == 'trainer':
        train_model(bus)
    elif args.mode == 'inference':
        model_inference(bus)
    elif args.mode == 'visualize':
        visualize(bus)
    elif args.mode == 'view_recording':
        view_recording(bus)
    elif args.mode == 'testing':
        testing(bus)

    print(f"Total time: {round(time.time() - start, 2)} seconds")

def set_public_states(bus: MessageBus):
    """publish the core most common public topics"""

    top_level = Path(__file__).resolve().parents[2]
    sim_data_path = top_level / "data" / "sim_data"
    model_data_path = top_level / "data" / "model_data"

    # set state for save directories
    bus.set_state("common/data_directory", PathMessage(sim_data_path))
    bus.set_state("common/model_directory", PathMessage(model_data_path))

def generate_data(bus: MessageBus):
    """generate and save data"""

    # 100 sims, 100 seconds, 100hz sim freq, 8hz GMM estimation. 338.96 seconds
    # 236 GMM approximations / second on laptop
    # 5184 sims covers every example with 5deg offsets. at 8hz thats 5hrs of data collection 
    manager = SimManager(bus, 10, 100)
    manager.batch_process()

def train_model(bus: MessageBus):
    
    # use train.py
    pass

def model_inference(bus: MessageBus):

    # use model.py to run at inference time
    # use visualization
    pass

def visualize(bus: MessageBus):

    manager = SimManager(bus, 10, 100, save_sim=False)
    observer = manager.run_single_simulation(0, 10)
    observer.visualize()

def view_recording(bus: MessageBus):

    name = '4'

    folder = bus.get_state("common/data_directory").path
    for file in folder.iterdir():
        if file.suffix == ".pkl" and name in str(file):
            observer: Observer = Observer.load(file)
            observer.visualize()
            break


def testing(bus: MessageBus):

    # testing stuff
    pass



if __name__ == "__main__":
    main()