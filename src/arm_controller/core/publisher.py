from .message_bus import MessageBus
from .message_types import Message

class Publisher:
    def __init__(self, bus: MessageBus, topic: str):
        self.bus = bus
        self.topic = topic

    def publish(self, msg: Message):
        self.bus.publish(self.topic, msg)