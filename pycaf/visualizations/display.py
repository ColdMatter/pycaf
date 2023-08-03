from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

from ..analysis import (
    Study
)


class Display(metaclass=ABCMeta):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def __call__(
        self,
        study: Study,
        **kwargs
    ) -> Tuple[Any, Any]:
        return NotImplementedError
