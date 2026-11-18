import time


class Timer:
    """Times operations and collects their durations keyed by description."""

    def __init__(self):
        self.timings = {}

    @staticmethod
    def print_header(title, header_length=50, newline="\n"):
        print(newline + "=" * header_length)
        print(title)
        print("=" * header_length)

    def time(self, description, operation):
        print(f"{description:<38}", end="")
        start = time.perf_counter()
        result = operation()
        end = time.perf_counter()
        duration = end - start
        print(f" {duration:>9.6f} s")
        self.timings[description] = duration
        return result

    def print_timings(self):
        self.print_header("TIMINGS")
        total_time = sum(self.timings.values())
        for operation, duration in self.timings.items():
            percentage = (duration / total_time) * 100
            print(f"{operation:<30} {duration:>8.3f} s ({percentage:>5.1f}%)")
        print(f"{'Total':<30} {total_time:>8.3f} s")
