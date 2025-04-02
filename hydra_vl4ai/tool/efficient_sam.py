import os
import cv2
import numpy as np
from ..util.misc import get_root_folder
from ._base import module_registry, BaseModel
from torchvision.transforms import ToTensor
import torch
import tensorneko_util as N


@module_registry.register("efficient_sam")
class EfficientSam(BaseModel):

    def __init__(self, gpu_number):
        super().__init__(gpu_number)
        if not os.path.exists(get_root_folder() / "pretrained_models" / "efficient_sam" / "efficientsam_s_gpu.jit"):
            self.prepare()
        self.model = torch.jit.load(str(get_root_folder() / "pretrained_models" / "efficient_sam" / "efficientsam_s_gpu.jit"), map_location=self.dev)
        self.to_tensor = ToTensor()

    @torch.no_grad()
    def forward(self, image: np.ndarray, bbox):
        left, lower, right, upper, confidence = bbox
        x0 = left + 10
        x1 = right - 10
        y0 = image.shape[0] - upper + 10
        y1 = image.shape[0] - lower - 10
        bbox = [x0, y0, x1, y1]
        bbox = torch.reshape(torch.tensor(bbox), [1, 1, 2, 2])
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.to_tensor(image)

        predicted_logits, predicted_iou = self.model(
            img_tensor[None, ...].to(self.dev),
            bbox.to(self.dev),
            bbox_labels.to(self.dev),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
            ):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou

    @classmethod
    def prepare(cls):
        path = get_root_folder() / "pretrained_models" / "efficient_sam"
        N.util.download_file("https://huggingface.co/spaces/yunyangx/EfficientSAM/resolve/main/efficientsam_s_gpu.jit", str(path))
