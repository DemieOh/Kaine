from typing import Optional, TextIO

import logging


def setup_logger(
        name: Optional[str] = 'kaine',
        level: int = logging.INFO,
        stream: Optional[TextIO] = None,
        format_spec: str = '\r[%(asctime)-15s] (%(filename)s:line %(lineno)d) %(name)s:%(levelname)s :: %(message)s',
        f: Optional[str] = None
    ) -> None:
    pass
