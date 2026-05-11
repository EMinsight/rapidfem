"""Tiny pub-sub bus for server → client WebSocket events.

Backend handlers call ``bus.publish({...})``; the ``/ws`` route hands each
subscriber a thread-safe queue and drains it on the WS write side.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[str]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=512)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, default=str)
        with self._lock:
            dead: list[queue.Queue[str]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)


BUS = EventBus()
