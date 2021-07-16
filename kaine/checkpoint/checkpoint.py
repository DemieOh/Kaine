from typing import BinaryIO, IO, Mapping, Optional, Union

import os
import logging
import torch

from torch.multiprocessing import Process, Lock


def enable_checkpoint_location(
        checkpoint_dir: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    ) -> None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(
        checkpoint: Mapping,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    ) -> None:
    torch.save(checkpoint, f)


def save_checkpoint_with_lock(
        checkpoint: Mapping,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        lock: Lock,
    ) -> None:
    lock.acquire()
    try:
        save_checkpoint(checkpoint, f)
    except Exception as e:
        logging.error(f'error occurs when saving checkpoint by "{e}"')
    finally:
        lock.release()


def save_checkpoint_asynchronously(
        checkpoint: Mapping,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        lock: Lock,
    ) -> None:
    Process(target=save_checkpoint_with_lock, args=(checkpoint, f, lock)).start()


def load_checkpoint(
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Mapping:
    device = 'cpu' if device is None else device
    return torch.load(f, map_location=device)


class CheckpointIO:

    def __init__(
            self,
            checkpoint_dir: Union[str, os.PathLike, BinaryIO, IO[bytes]],
            save_asynchronously: bool = False
        ):
        self.checkpoint_dir = checkpoint_dir
        self.is_async = save_asynchronously
        self.io_function = save_checkpoint if not self.is_async else save_checkpoint_asynchronously
        self.lock = Lock()

        enable_checkpoint_location(self.checkpoint_dir)

    def save(
            self,
            checkpoint: Mapping,
            f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        ) -> None:

        if os.path.sep not in f:
            logging.info(f'it looks like `{f}` does not including the parent dirs, therefore it will be saved in the preset dir')
            f = os.path.join(self.checkpoint_dir, f)

        try:
            if self.is_async:
                self.io_function(checkpoint, f, self.lock)
            else:
                self.io_function(checkpoint, f)
                logging.debug(f'saving checkpoint at {f} successfully')
        except Exception as e:
            logging.error(f'error occurs when writing checkpoint on disk by "{e}"')

    def load(
            self,
            f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
            device: Optional[Union[str, torch.device]] = None,
        ) -> Mapping:

        if os.path.sep not in f:
            logging.info(f'it looks like `{f}` does not including the parent dirs, therefore it will be loaded from checkpoint in the preset dir')
            f = os.path.join(self.checkpoint_dir, f)

        try:
            c = load_checkpoint(f, device=device)
            logging.debug(f'loading checkpoint from {f} successfully')
            return c
        except Exception as e:
            logging.error(f'error occurs when reading checkpoint from disk by "{e}"')
