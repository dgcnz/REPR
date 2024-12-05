import subprocess
import threading
import time


class MemTracker:
    def __init__(self):
        """
        Initializes the memory tracker.

        Args:
            frequency (float): How often to poll GPU memory (in seconds).
        """
        self.polling = False
        self._thread = None
        self.frequency = None
        self.mem_history = []

    def _poll_memory(self):
        """
        Polls GPU memory usage at regular intervals and updates the peak memory usage.
        """
        while self.polling:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                current_memory = int(result.stdout.strip())
                self.mem_history.append(current_memory)
            except ValueError:
                pass  # Ignore invalid output
            time.sleep(self.frequency)

    def start_polling(self, frequency: float = 0.1):
        """Starts the polling process in a background thread."""
        if self.polling:
            print("Polling is already running.")
            return
        self.frequency = frequency
        self.polling = True
        self.mem_history = []
        self._thread = threading.Thread(target=self._poll_memory, daemon=True)
        self._thread.start()

    def stop_polling(self):
        """Stops the polling process."""
        if not self.polling:
            print("Polling is not running.")
            return
        self.polling = False
        self._thread.join()

    def get_max_mem(self):
        """
        Returns the maximum memory usage recorded during the polling period.

        Returns:
            int: Peak memory usage in MB.
        """
        return max(self.mem_history)
