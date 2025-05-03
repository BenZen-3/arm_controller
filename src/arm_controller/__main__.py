import argparse
import time


from .core.message_bus import MessageBus
from .core.subscriber import Subscriber
from .core.publisher import Publisher
from .core.message_types import Message, NumberMessage, ListMessage, StringMessage
from .simulation.sim_manager import SimManager

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
    publish_public_topics(bus)

    if args.mode == 'generator':
        generate_data()
    elif args.mode == 'trainer':
        train_model()
    elif args.mode == 'inference':
        model_inference()
    elif args.mode == 'playback':
        playback()
    elif args.mode == 'testing':
        testing()

    print(f"Total time: {round(time.time() - start, 2)} seconds")

def publish_public_topics(bus: MessageBus):
    """publish the core most common public topics"""
    
    # publsih data save directory
    data_save_dir_pub = Publisher(bus, "data_save_dir")
    msg = StringMessage("INSERT DATA DIRECTORY HERE")
    data_save_dir_pub.publish(msg)

    # publish model save directory
    model_save_dir_pub = Publisher(bus, "model_save_dir")
    msg = StringMessage("INSERT MODEL DIRECTORY HERE")
    model_save_dir_pub.publish(msg)


def generate_data():

    manager = SimManager()
    manager.generate_data()

    

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