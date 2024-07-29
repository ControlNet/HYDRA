import pathlib
from PIL import Image
from typing import Union
import requests
import torchvision
import json
import io


def load_image(path):
    to_tensor = torchvision.transforms.ToTensor()
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = to_tensor(image)
    else:
        image = Image.open(path)
        image = to_tensor(image)
    return image


def load_image_from_bytes(data: bytes):
    image = Image.open(io.BytesIO(data))
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(image)


def load_json(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_root_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


def get_hydra_root_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_statement_variable(selected_code, local_variables):
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


def get_description_from_executed_variable_list(executed_variable_list, local_variables) -> list[str]:
    from ..execution.image_patch import ImagePatch
    description = []
    for variable_name in executed_variable_list:
        one_variable = local_variables[variable_name]
        description.append(f'{variable_name}: {one_variable}')
        if isinstance(one_variable, ImagePatch):
            description[-1] += f' patch name: {one_variable.image_name};'
    return description

