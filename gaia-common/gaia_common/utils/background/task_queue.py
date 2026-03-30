"""
background/task_queue.py

Handles task queue management for GAIA background processing.
Manages adding tasks, saving/loading task status, and retry logic.
"""

import queue
import threading
import logging
from typing import Optional, Dict

logger = logging.getLogger("GAIA.TaskQueue")

class TaskQueue:
    """
    Thread-safe FIFO task queue for background job execution.
    Supports tagging and dynamic prioritization in future extensions.
    """

    def __init__(self):
        self.queue = queue.Queue()
        self.lock = threading.Lock()

    def add_task(self, task: Dict):
        """
        Add a task to the queue.

        Args:
            task (dict): Task with at least a 'type' field.
        """
        with self.lock:
            self.queue.put(task)
            logger.info(f"🧾 Task queued: {task.get('type')} :: {task.get('tag', '[untagged]')}")

    def pop_next_task(self) -> Optional[Dict]:
        """
        Retrieve and remove the next task from the queue.

        Returns:
            dict or None
        """
        with self.lock:
            if self.queue.empty():
                return None
            return self.queue.get()

    def is_empty(self) -> bool:
        """Check if the queue is currently empty."""
        return self.queue.empty()

    def size(self) -> int:
        """Return the number of tasks currently in the queue."""
        return self.queue.qsize()
