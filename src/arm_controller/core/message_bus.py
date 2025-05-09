from collections import defaultdict
from typing import Callable, Dict, List, Optional
from copy import deepcopy

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

    def branch_bus(self) -> "MessageBus":
        """Returns a new MessageBus with a copy of the current state but no subscribers."""
        new_bus = MessageBus()
        new_bus._state_store = deepcopy(self._state_store)
        return new_bus

    def list_topics(self) -> List[str]:
        """
        Returns a list of all topics that currently have subscribers.
        """
        return list(self.subscribers.keys())

    def list_states(self) -> List[str]:
        """
        Returns a list of all topics that have stored state (via set_state).
        """
        return list(self._state_store.keys())
