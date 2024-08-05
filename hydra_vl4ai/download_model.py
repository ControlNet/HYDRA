import tensorneko_util as N
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
args = parser.parse_args()

from .util.config import Config
Config.base_config_path = args.base_config
Config.model_config_path = args.model_config
Config.base_config["debug"] = False
from .util.console import console
from .tool import module_registry


def prepare_models():
    model_config = N.read.yaml(Config.model_config_path)
    for _, model_names in model_config["cuda"].items():
        for model_name in model_names:
            ModelClass = module_registry[model_name]
            ModelClass.prepare()


if __name__ == '__main__':
    with console.status("[bold green]Downloading models..."):
        prepare_models()
