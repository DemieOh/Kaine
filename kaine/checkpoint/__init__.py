from collections import OrderedDict
from collections.abc import Mapping


class Serializable:

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    @staticmethod
    def load_state_dict(state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError(f'state_dict should be a mapping, but given {type(state_dict)}')
