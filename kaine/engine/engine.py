from typing import Callable

import logging
import torch.utils.data


class Engine:

    def __init__(
            self,
            process_fn: Callable,
        ) -> None:
        self.logger = logging.getLogger('.'.join([__name__, self.__class__.__name__]))
        self.process_fn = process_fn
        if self.process_fn is None:
            raise ValueError('engine must be given a function to process in order to run')

    def __call__(
            self,
            dataloader: torch.utils.data.DataLoader,
        ) -> None:
        for index, batch in enumerate(dataloader):
            self.process_fn(batch)
