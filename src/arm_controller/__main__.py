import argparse
import time


from .core.message_bus import MessageBus
from .core.subscriber import Subscriber
from .core.publisher import Publisher
from .core.message_types import Message, NumberMessage, ListMessage, StringMessage
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
    
    # set state for data save directory
    msg = StringMessage("INSERT DATA DIRECTORY HERE")
    bus.set_state("common/data_directory", msg)

    # set state for model save directory
    msg = StringMessage("INSERT MODEL DIRECTORY HERE")
    bus.set_state("common/model_directory", msg)


def generate_data(bus: MessageBus):

    manager = SimManager(bus, 1, 10)
    # manager.run_single_simulation()
    manager.batch_process()

    

    # create message bus
    # create simulation node
    # create data synthesis node
    # connect them to the bus
    # save the data
    pass

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