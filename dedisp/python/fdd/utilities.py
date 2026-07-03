import time


class Timer:
    def __init__(self):
        self.is_running: bool = False
        self.time_sum: float = 0.0
        self.time_start: float = 0.0

    def start(self):
        if not self.is_running:
            self.time_start = time.time()
            self.is_running = True

    def pause(self):
        if self.is_running:
            now = time.time()
            self.time_sum += now - self.time_start

            self.is_running = False

    def reset(self):
        self.time_sum = 0.0
        self.is_running = False

    def duration(self):
        return self.time_sum
