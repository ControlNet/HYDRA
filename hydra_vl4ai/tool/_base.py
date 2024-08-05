import abc
from tensorneko_util.util import Registry


class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}'

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def prepare(cls):
        """This method should be used to download the model and prepare it for use. It should be a classmethod."""
        pass


module_registry: Registry[BaseModel] = Registry()
