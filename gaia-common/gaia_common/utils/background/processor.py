"""
background/processor.py

BackgroundProcessor handles idle-time and overnight-time task processing.
"""

import threading
import time
import logging

from app.utils.background.task_queue import TaskQueue
from app.utils.background.background_tasks import BackgroundTask
from app.utils.background.idle_monitor import IdleMonitor
from app.cognition.initiative_handler import gil_check_and_generate

logger = logging.getLogger("GAIA.BackgroundProcessor")

class BackgroundProcessor:
    """
    Runs in the background and monitors idle conditions to process queued tasks.
    Coordinates summarization, embedding, artifact generation, code analysis, and initiative prompting.
    """

    def __init__(self, config):
        self.config = config
        self.task_queue = TaskQueue()
        self.task_handler = BackgroundTask()
        self.idle_monitor = IdleMonitor()
        self.thread = None
        self.running = False

        # Injected externally after init
        self.ai_manager = None
        self.conversation_manager = None
        self.vector_store_manager = None
        self.vector_store = None
        self.doc_processor = None

    def start(self):
        """Starts the background processor in a new thread."""
        if not self.thread:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            logger.info("🌀 Background thread launched.")

    def stop(self):
        """Signals the thread to stop gracefully."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info("🛑 Background thread stopped.")

    def run(self):
        """
        Background loop that checks for idle time and processes tasks.
        """
        logger.info("📡 Background processor running.")

        while self.running:
            try:
                if not self.idle_monitor.is_system_idle():
                    time.sleep(5)
                    continue

                task = self.task_queue.pop_next_task()
                if task:
                    logger.info(f"📥 Running background task: {task.get('type')} :: {task.get('tag')}")
                    task_result = self.task_handler.process_conversation_task(task)
                    logger.info(f"✅ Task completed: {task_result}")
                else:
                    # Periodic idle checks like reflection or summaries
                    if self.ai_manager and self.ai_manager.self_reflection:
                        self.ai_manager.self_reflection.idle_time_check()

                    # Initiative prompt check using config-based pathing
                    idle_minutes = self.idle_monitor.get_idle_minutes()
                    gil_check_and_generate(user_idle_minutes=idle_minutes, config=self.config)
                    if initiative_message and self.conversation_manager:
                        self.conversation_manager.post_ai_message(initiative_message)

                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Error in background processor: {e}", exc_info=True)
                time.sleep(15)
