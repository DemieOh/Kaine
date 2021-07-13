import logging

from torch.multiprocessing import Process, Lock


def this_fails(foo: int):
    return foo / 0


def save_asynchronously(foo, lock: Lock):
    lock.acquire()
    try:
        print(foo)
    except RuntimeError as e:
        logging.error(f'error occurs when saving checkpoint by "{e}"')
    finally:
        lock.release()


def factory_fn(lock):
    def __save(foo):
        return Process(target=save_asynchronously, args=(foo, lock))
    return __save


if __name__ == '__main__':
    lock = Lock()
    proc = factory_fn(lock)

    for idx in range(10):
        p = proc(idx)
        p.start()
        # p.join()
