import asyncio
import json
import os
import requests
import time
import numpy as np
from PIL import Image
import tensorneko_util as N
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--query", type=str, required=True)
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--result_folder", type=str, default="./result")
args = parser.parse_args()

from hydra_vl4ai.util.config import Config

Config.base_config_path = args.base_config
Config.model_config_path = args.model_config

from hydra_vl4ai.agent.hydra import HydraNoRL
from hydra_vl4ai.util.console import logger, console
from hydra_vl4ai.util.misc import wait_until_loaded


async def main():
    with console.status("[bold green]Connect to HYDRA executor...") as status:
        wait_until_loaded(f"http://localhost:{Config.base_config['executor_port']}")
    hydra = HydraNoRL()

    # with open(args.image, "rb") as f:
    #     image_buffer = f.read()

    img = Image.open(args.image)
    img = img.convert("RGB")

    img_array = np.array(img)

    return img_array
    result = await hydra(image_buffer, args.query)
    logger.info(f"Query: {args.query} Answer: {result}")


if __name__ == "__main__":
    asyncio.run(main=main())
