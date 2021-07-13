from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim


def load_batch(
        batch: Sequence[torch.Tensor],
        device: Optional[Union[str, torch.Tensor]] = None,
        non_blocking: bool = False,
    ) -> Tuple[Union[torch.Tensor, Sequence, str, bytes], ...]:
    return (b.to(device=device, non_blocking=non_blocking) for b in [*batch]) if device is not None else batch


def update(
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        return_form: Callable = lambda i, t, o, l: (o, t)
    ) -> Callable:
    def __update(batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        inputs, targets = load_batch(batch, device=device, non_blocking=non_blocking)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        return return_form(inputs, targets, outputs, loss)
    return __update


def evaluate(
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        return_form: Callable = lambda i, t, o: (o, t)
    ) -> Callable:
    def __evaluate(batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            inputs, targets = load_batch(batch, device=device, non_blocking=non_blocking)
            outputs = model(inputs)
        return return_form(inputs, targets, outputs)
    return __evaluate
