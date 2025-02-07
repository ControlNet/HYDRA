import argparse

import tensorneko_util as N

from evaluation.grounding_eval import batch_iou_2d
from evaluation.vqa_eval import GQAeval
from hydra_vl4ai.util.console import console, logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    with console.status("[bold green]Loading JSON...") as status:
        result = N.io.read.json(args.result_path)
        evaluator = GQAeval()
        status.update("[bold green]Evaluating...")

        match args.dataset:
            case "okvqa":
                score = evaluator.accuracy_one_set([each["result"] for each in result],
                    [each["ground_truth"] for each in result])
            case "refcoco" | "refcoco+":
                score = batch_iou_2d(result)
            case _:
                raise ValueError(f"Dataset {args.dataset} is not supported.")
        logger.info(f"Score: {score}")
