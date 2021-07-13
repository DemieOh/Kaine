from typing import Dict, Union
import copy
import logging

import torch.nn as nn
import torch.optim as optim



class CheckpointModule:

    def __init__(
            self,
            checkpoint: Dict[str, Union[str, int, list, dict, nn.Module, optim.Optimizer]]
        ) -> None:
        self.checkpoint = checkpoint
        self.latest_snapshot = None

    def snapshot(
            self,
        ) -> bool:
        if self.latest_snapshot is not None:
            logging.error('the snapshot already exists, therefore, this action cannot be done')
            return False

        try:
            self.latest_snapshot = {k: copy.deepcopy(v) for k, v in self.checkpoint.items()}
            return True
        except Exception as e:
            logging.error(f'error occurs when trying to make snapshot by {e}')
