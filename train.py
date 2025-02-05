import asyncio
import json
import os
import requests
import time
import tensorneko_util as N
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--result_folder", type=str, default="./result")
parser.add_argument("--dqn_config", type=str, required=True)
args = parser.parse_args()

from hydra_vl4ai.util.config import Config
Config.base_config_path = args.base_config
Config.dqn_config_path = args.dqn_config
Config.model_config_path = args.model_config

from hydra_vl4ai.agent.hydra import HydraNoRL, HydraTrainingRL
from hydra_vl4ai.util.console import logger, console
from hydra_vl4ai.util.misc import wait_until_loaded
import exp_datasets


async def main():
    with console.status("[bold green]Connect to HYDRA executor...") as status:
        wait_until_loaded(f"http://localhost:{Config.base_config['executor_port']}")
    hydra = HydraTrainingRL()

    match Config.base_config["dataset"]:
        case "gqa":
            dataset = exp_datasets.GQA(
                args.data_root
            )
        case "okvqa":
            dataset = exp_datasets.OKVQA(
                args.data_root
            )
        case "aokvqa":
            # TODO: Not tested yet
            # dataset = exp_datasets.AOKVQA(
            #     f"{args.data_root}/aokvqa", 
            #     "val", f"{args.data_root}/coco", version="v1p0"
            # )
            raise NotImplementedError("AOKVQA is not implemented yet")
        case "refcoco":
            dataset = exp_datasets.Refcoco(args.data_root)
        case "refcoco+":
            dataset = exp_datasets.Refcoco(args.data_root)
        case _:
            raise ValueError("Invalid dataset")
        
    # output path
    Path(args.result_folder).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.result_folder) / f"result_{Config.base_config['dataset']}.jsonl"

    cum_reward=0 # TODO:modify 

    for epoch_idx_ in range(Config.dqn_config["training_epoch"]):       
        for i, (image_path, datum_id, query, ground_truth) in enumerate(dataset):
            
            logger.info(f"Processing {i+1}/{len(dataset)}")
            with open(image_path, "rb") as f:
                image_buffer = f.read()
            result = await hydra(image_buffer, query, ground_truth)
            logger.info(f"Query: {query} Answer: {result}")

            with open(save_path, "a") as f:
                f.write(json.dumps({
                    "datum_id": datum_id,
                    "query": query,
                    "ground_truth": ground_truth,
                    "result": result
                }) + "\n")
                f.flush()

            # training log info
            if hydra.controller.obs_no % hydra.controller.train_log_interval == 0:
                mean_reward = np.mean(hydra.controller.reward_window)
                cum_reward = 0.99*cum_reward + 0.01*mean_reward
                logger.info('---Current step:{}-----Mean Reward:{:.2f}----Cumulative Reward:{:.2f}'.format(hydra.controller.obs_no, mean_reward, cum_reward))
            
            if hydra.controller.save_model_obs_num > hydra.controller.save_interval \
                and hydra.controller.best_cum_reward < cum_reward:

                # update best cummulated reward
                hydra.controller.best_cum_reward = cum_reward

                # save model
                hydra.controller.rl_agent_model.save_model(os.path.join(hydra.controller.model_save_path, Config.dqn_config["model_name"]))
                
                hydra.controller.save_model_obs_num = 0


if __name__ == "__main__":
    asyncio.run(main=main())
