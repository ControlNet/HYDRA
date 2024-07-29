import asyncio
import json
import os
import requests
import time
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

from hydra.util.config import Config

Config.base_config_path = args.base_config
Config.model_config_path = args.model_config

from hydra.agent.hydra import HydraNoRL
from hydra.util.console import logger, console


def wait_until_loaded():
    while True:
        try:
            response = requests.get(f"http://localhost:{Config.base_config['executor_port']}/is_loaded")
            if response.json()["result"]:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.1)


async def main():
    with console.status("[bold green]Connect to HYDRA executor...") as status:
        wait_until_loaded()
    hydra = HydraNoRL()

    with open(args.image, "rb") as f:
        image_buffer = f.read()
    result = await hydra(image_buffer, args.query)
    logger.info(f"Query: {args.query} Answer: {result}")


if __name__ == "__main__":
    asyncio.run(main=main())
