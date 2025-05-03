from .message_bus import MessageBus
from .message_types import Message
from typing import Callable

class Subscriber:
    def __init__(self, bus: MessageBus, topic: str, callback: Callable[[Message], None]):
        self.bus = bus
        self.bus.subscribe(topic, callback)