import tensorneko_util as N
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
# list of extra packages
parser.add_argument("--extra_packages", type=str, nargs="*", default=[])
args = parser.parse_args()

from .util.config import Config
Config.base_config_path = args.base_config
Config.model_config_path = args.model_config
Config.debug = False
from .util.console import console
from .tool import module_registry


def prepare_models():
    for package in args.extra_packages:
        __import__(package)

    model_config = N.read.yaml(Config.model_config_path)
    for _, model_names in model_config["cuda"].items():
        for model_name in model_names:
            ModelClass = module_registry[model_name]
            ModelClass.prepare()


if __name__ == '__main__':
    with console.status("[bold green]Downloading models..."):
        prepare_models()
