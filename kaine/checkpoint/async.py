import logging

from torch.multiprocessing import Process, Lock


def this_fails(x):
    return x / 0  # for debugging, raise division by zero error


def save_asynchronously(foo, lock: Lock):
    lock.acquire()
    try:
        print(foo)
    except RuntimeError as e:
        logging.error(f'error occurs when saving checkpoint by "{e}"')
    finally:
        lock.release()


if __name__ == '__main__':
    pass
