import asyncio
from dotenv import load_dotenv

load_dotenv()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--share", default=False, action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

from hydra_vl4ai.util.config import Config

Config.base_config_path = args.base_config
Config.model_config_path = args.model_config
Config.debug = args.debug

from hydra_vl4ai.agent.webui import HydraNoRLWeb
from hydra_vl4ai.util.console import console
from hydra_vl4ai.util.misc import wait_until_loaded


async def main():
    with console.status("[bold green]Connect to HYDRA executor..."):
        wait_until_loaded(f"http://localhost:{Config.base_config['executor_port']}")

    hydra = HydraNoRLWeb()
    hydra.gradio_app.launch(share=args.share)


if __name__ == "__main__":
    asyncio.run(main())
