from typing import Callable

import torch.utils.data


class Engine:

    def __init__(
            self,
            process_fn: Callable,
        ):
        self.process_fn = process_fn

    def __call__(
            self,
            dataloader: torch.utils.data.DataLoader,
        ):
        for index, batch in enumerate(dataloader):
            self.process_fn(batch)
