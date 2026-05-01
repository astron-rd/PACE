import time

# Dictionary to store all timings
timings = dict()


def timeit(description, operation):
    print(f"{description:<38}", end="")
    start = time.time()
    result = operation()
    end = time.time()
    duration = end - start
    print(f" {duration:>9.6f} s")
    timings[description] = duration
    return result


def main():
    print("DEDISP MAIN")
