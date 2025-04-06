import pathlib
import time
from PIL import Image
from typing import Any, Union
import requests
from torch import Tensor
from torchvision.transforms import functional as T
import json
import io


def load_image(path: str) -> Tensor:
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = T.to_tensor(image)
    else:
        image = Image.open(path)
        image = T.to_tensor(image)
    return image


def load_image_from_bytes(data: bytes) -> Tensor:
    image = Image.open(io.BytesIO(data))
    return T.to_tensor(image)


def load_json(path: Union[str, pathlib.Path]) -> list | dict:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_root_folder() -> pathlib.Path:
    return pathlib.Path.cwd()


def get_hydra_root_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_statement_variable(selected_code: str, local_variables: dict[str, Any]) -> list[str]:
    """
    INPUT:
        selected_code: Selected execution code
        local_variables: list of LOCAL VARIABLE (delete after each query finished)

    OUTPUT:
        selected code and probability
    """
    executed_variable_list = []

    for one_row_code in selected_code.split('\n'):
        if ' = ' in one_row_code: # has variable
            variable_name = one_row_code.split(' = ')[0] # get variable name --str type
            if variable_name in local_variables:
                executed_variable_list.append(variable_name) # append local variable into list.

    return executed_variable_list


def get_description_from_executed_variable_list(executed_variable_list: list[str], local_variables: dict[str, Any]) -> list[str]:
    # run in executor only
    from ..execution.image_patch import ImagePatch
    description = []
    for variable_name in executed_variable_list:
        one_variable = local_variables[variable_name]
        description.append(f'{variable_name}: {one_variable}')
        if isinstance(one_variable, ImagePatch):
            description[-1] += f', patch name: {one_variable.image_name}'
    return description


def wait_until_loaded(executor_url: str) -> None:
    while True:
        try:
            response = requests.get(f"{executor_url}/is_loaded")
            if response.json()["result"]:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.1)
