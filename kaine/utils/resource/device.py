from typing import Mapping

import os
import logging
import subprocess
import torch

#TODO: Support ROCm


def auto_device() -> torch.device:
    logging.debug(f'trying to search for suitable device among all available device')
    try:
        return torch.device(f'cuda:{CUDAInfo().nvidia_gpu_memory_usage[0][0]}')
    except RuntimeError as e:
        if not torch.cuda.is_available():
            logging.warning('it looks like can not find GPU on your device, therefore device switched into cpu automatically')
            return torch.device('cpu')
        else:
            logging.error(f'error occurs when searching device for allocating by "{e}"')
            raise e


class CUDAInfo:

    def __init__(self, sorting: bool = True) -> None:

        if 'CUDA_DEVICE_ORDER' not in os.environ or 'PCI_BUS_ID' != os.environ['CUDA_DEVICE_ORDER']:
            warn = 'It`s recommended to set ``CUDA_DEVICE_ORDER`` to be ``PCI_BUS_ID`` by ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``; ' \
                   'Otherwise, it`s not guaranteed that the GPU index from PyTorch to be consistent the ``nvidia-smi`` results.'
            logging.debug(warn)

        self.nvidia_gpu_memory_usage = self.__util__(query='--query-gpu=memory.used')
        if sorting:
            self.nvidia_gpu_memory_usage = sorted(self.nvidia_gpu_memory_usage.items(), key=lambda item: item[1])

    def __util__(self, query: str) -> Mapping:

        def shift(memory_size: str):
            size, suffix = memory_size.split(' ')
            size = int(size)
            lookup = {
                'KB': size << 10, 'KiB': size << 10,
                'MB': size << 20, 'MiB': size << 20,
                'GB': size << 30, 'GiB': size << 30,
            }
            return lookup[suffix]

        res = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader']).decode().strip().split('\n')
        return {x: shift(y) for x, y in enumerate(res)}
