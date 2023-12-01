from time import sleep

def wait_until(func, interval = 1, time_limit = None):
    total_time = 0

    while (not func()) and (time_limit is None or total_time < time_limit):
        sleep(interval)

        total_time += interval
