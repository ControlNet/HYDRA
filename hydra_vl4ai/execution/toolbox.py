import inspect
import tensorneko_util as N
from tensorneko_util.util import Singleton

from ..tool import module_registry
from ..util.config import Config
from ..util.console import logger, console


@Singleton
class Toolbox:

    def __init__(self) -> None:
        self.consumers = dict()
        self.inited = False

    def init(self):
        if self.inited:
            return

        model_config = N.read(Config.model_config_path)
        for cuda_id, model_names in model_config["cuda"].items():
            for model_name in model_names:
                ModelClass = module_registry[model_name]
                self.consumers[model_name] = self.make_fn(
                    ModelClass, model_name, cuda_id)
        self.inited = True

    @staticmethod
    def make_fn(model_class, model_name, gpu_number):

        model_instance = model_class(gpu_number=gpu_number)

        def _function(*args, **kwargs):
            if model_class.to_batch:
                # Batchify the input. Model expects a batch. And later un-batchify the output.
                args = [[arg] for arg in args]
                kwargs = {k: [v] for k, v in kwargs.items()}

                # The defaults that are not in args or kwargs, also need to listify
                full_arg_spec = inspect.getfullargspec(model_instance.forward)
                if full_arg_spec.defaults is None:
                    default_dict = {}
                else:
                    default_dict = dict(
                        zip(full_arg_spec.args[-len(full_arg_spec.defaults):], full_arg_spec.defaults))
                non_given_args = full_arg_spec.args[1:][len(args):]
                non_given_args = set(non_given_args) - set(kwargs.keys())
                for arg_name in non_given_args:
                    kwargs[arg_name] = [default_dict[arg_name]]

            try:
                out = model_instance.forward(*args, **kwargs)
                if model_class.to_batch:
                    out = out[0]
            except Exception as e:
                logger.error(f'Error in {model_name} model:', e)
                # stack trace
                console.print_exception(show_locals=True)
                out = None
            return out

        return _function

    def forward(self, model_name, *args, queues=None, **kwargs):
        """
        Sends data to consumer (calls their "forward" method), and returns the result
        """
        error_msg = f'No model named {model_name}. ' \
                    'The available models are: {}. Make sure to activate it in the configs files'
        try:
            out = self.consumers[model_name](*args, **kwargs)
        except KeyError as e:
            raise KeyError(error_msg.format(list(self.consumers.keys()))) from e
        return out


forward = Toolbox.forward
