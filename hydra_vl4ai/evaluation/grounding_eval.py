from typing import Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def iou_2d(proposal: Union[Tensor, ndarray], target: Union[Tensor, ndarray]) -> Tensor:
    """
    Calculate 2D IOU for M proposals with N targets.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The proposals array with shape [M, 4]. The 4
            columns represents x1, y1, x2, y2.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The targets array with shape [N, 4]. The 4 columns
            represents x1, y1, x2, y2.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is ndarray:
        proposal = torch.tensor(proposal)

    if type(target) is ndarray:
        target = torch.tensor(target)

    proposal_x1 = proposal[:, 0]
    proposal_y1 = proposal[:, 1]
    proposal_x2 = proposal[:, 2]
    proposal_y2 = proposal[:, 3]

    target_x1 = target[:, 0].unsqueeze(0).T
    target_y1 = target[:, 1].unsqueeze(0).T
    target_x2 = target[:, 2].unsqueeze(0).T
    target_y2 = target[:, 3].unsqueeze(0).T

    inner_x1 = torch.maximum(proposal_x1, target_x1)
    inner_y1 = torch.maximum(proposal_y1, target_y1)
    inner_x2 = torch.minimum(proposal_x2, target_x2)
    inner_y2 = torch.minimum(proposal_y2, target_y2)

    area_proposal = (proposal_x2 - proposal_x1) * (proposal_y2 - proposal_y1)
    area_target = (target_x2 - target_x1) * (target_y2 - target_y1)

    inter_x = torch.clamp(inner_x2 - inner_x1, min=0.)
    inter_y = torch.clamp(inner_y2 - inner_y1, min=0.)
    inter = inter_x * inter_y

    union = area_proposal + area_target - inter

    return inter / union


def process_grounding_result(x: str):
    return eval(x)["final_answer"][0][:4]


def batch_iou_2d(result):
    # we follow ViperGPT to filter out the None results
    remove_index = []
    for i in range(len(result)):
        if result[i]["result"] is None:
            remove_index.append(i)

    # apply to the result
    for i in remove_index[::-1]:
        del result[i]

    proposals = [process_grounding_result(each["result"]) for each in result]
    targets = [each["ground_truth"] for each in result]

    scores = []
    for i in range(len(proposals)):
        scores.append(iou_2d(np.array(proposals[i])[None], np.array(targets[i])[None])[0, 0])

    return np.mean(scores)
