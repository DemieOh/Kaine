from typing import BinaryIO, IO, Union

import logging
import os
import torch


from torch.multiprocessing import Process, Lock


def enable_checkpoint_location(
        checkpoint_dir: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    ) -> None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


def save_checkpoint(
        checkpoint,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    ) -> None:
    torch.save(checkpoint, f)


def save_checkpoint_with_lock(
        checkpoint,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        lock: Lock
    ) -> None:
    lock.acquire()
    try:
        save_checkpoint(checkpoint, f)
    except Exception as e:
        logging.error(f'error occurs when saving checkpoint by "{e}"')
    finally:
        lock.release()


def save_checkpoint_asynchronously(
        checkpoint,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        lock: Lock
    ) -> None:
    Process(target=save_checkpoint_with_lock, args=(checkpoint, f, lock)).start()
