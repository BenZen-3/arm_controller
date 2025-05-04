import argparse
import time
from pathlib import Path

from .core.message_bus import MessageBus
# from .core.subscriber import Subscriber
# from .core.publisher import Publisher
from .core.message_types import PathMessage
from .simulation.sim_manager import SimManager

"""
todo:
    message bus needs to be simulation-based. If there is a single message bus with multiprocess it will get ugly I think


"""


def main():
    """
    main program start
    """

    parser = argparse.ArgumentParser(description="Arm controller Package")
    parser.add_argument(
        '--mode', 
        choices=['generator', 'trainer', 'inference', 'playback', "testing"], 
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
        train_model()
    elif args.mode == 'inference':
        model_inference()
    elif args.mode == 'playback':
        playback()
    elif args.mode == 'testing':
        testing()

    print(f"Total time: {round(time.time() - start, 2)} seconds")

def set_public_states(bus: MessageBus):
    """publish the core most common public topics"""

    top_level = Path(__file__).resolve().parent.parent.parent # jank
    sim_data_path = top_level / "data" / "sim_data"
    model_data_path = top_level / "data" / "model_data"

    # set state for save directories
    bus.set_state("common/data_directory", PathMessage(sim_data_path))
    bus.set_state("common/model_directory", PathMessage(model_data_path))


def generate_data(bus: MessageBus):

    manager = SimManager(bus, 1, 10)
    manager.run_single_simulation(0, 10, 100)

def train_model():
    
    # use train.py
    pass

def model_inference():

    # use model.py to run at inference time
    # use visualization
    pass

def playback():

    # use visualization tools to playback a recording
    pass

def testing():

    # testing stuff
    pass



if __name__ == "__main__":
    main()