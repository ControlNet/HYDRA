import abc

import numpy as np


class Controller(abc.ABC):

    @abc.abstractmethod
    def __call__(self, instructions: list[str], probs: np.ndarray) -> str:
        pass


class ControllerLLM(Controller):

    def __call__(self, instructions: list[str], probs: np.ndarray) -> str:
        return instructions[np.argmax(probs)]
