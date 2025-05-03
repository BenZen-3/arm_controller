from collections import defaultdict
from typing import Callable, Dict, List, Any, Optional

from .message_types import Message

class MessageBus:
    """All messaages go through da bus! Two types: push (sub/sub), pull(set_state, get_state)"""
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = defaultdict(list)
        # subscribers is a Dict with str keys, callable values that take messages and return nothing

        self._state_store: Dict[str, Message] = {}  
        # topic -> latest Message for sticky topics

    def publish(self, topic: str, msg: Message):
        for callback in self.subscribers[topic]:
            callback(msg)

    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        self.subscribers[topic].append(callback)

    def set_state(self, topic: str, msg: Message):
        self._state_store[topic] = msg

    def get_state(self, topic: str) -> Optional[Message]:
        return self._state_store.get(topic, None)

