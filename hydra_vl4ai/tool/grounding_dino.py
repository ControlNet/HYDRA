import os
import numpy as np
import torch

from ..util.config import Config
from ..util.console import logger
from ..util.misc import get_root_folder
from ._base import BaseModel, module_registry
import tensorneko_util as N
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.models import build_model as build_grounding_dino
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T
        

@module_registry.register("grounding_dino")
class GroundingDino(BaseModel):

    def __init__(self, gpu_number=0):

        super().__init__(gpu_number)
        path_checkpoint = str(get_root_folder() / "pretrained_models" / "grounding_dino" / "groundingdino_swint_ogc.pth")
        if not os.path.exists(path_checkpoint):
            self.prepare()
        config_file = str(get_root_folder() / "module_repos" / "Grounded-Segment-Anything" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
        args = SLConfig.fromfile(config_file) 
        args.device = self.dev
        self.deivce = self.dev
        self.gd = build_grounding_dino(args)

        checkpoint = torch.load(path_checkpoint, map_location='cpu')
        log = self.gd.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        logger.info("Model loaded from {} \n => {}".format(path_checkpoint, log))
        self.gd.eval()

    def image_transform_grounding(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None) # 3, h, w
        return init_image, image

    def transfer_boxes_format(self, boxes, height, width):
        boxes = boxes * torch.Tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        transfered_boxes = []
        for i in range(len(boxes)):
            box = boxes[i]
            transfered_box = [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]]
            transfered_boxes.append(transfered_box)
        
        transfered_boxes = np.array(transfered_boxes)
        return transfered_boxes
        
    @torch.no_grad()
    def forward(self, input_image, grounding_caption, box_threshold=None, text_threshold=0.25):
        '''
            return:
                annotated_frame:nd.array
                transfered_boxes: nd.array [N, 4]: [[x0, y0], [x1, y1]]
        '''
        if box_threshold is None:
            box_threshold = Config.base_config["grounding_dino_threshold"]
        input_image = np.asarray(input_image.permute(1,2,0)*255, dtype=np.uint8)
        height, width, _ = input_image.shape

        img_pil = Image.fromarray(input_image)
        re_width, re_height = img_pil.size
        _, image_tensor = self.image_transform_grounding(img_pil)
        # print('image_tensor shape: ', image_tensor.shape)
        # print(image_tensor)

        # run grounidng
        boxes, confidences, phrases = predict(self.gd, image_tensor, grounding_caption, box_threshold, text_threshold, device=self.deivce)

        # annotated_frame = annotate(image_source=np.asarray(img_pil), boxes=boxes, logits=logits, phrases=phrases)[:, :, ::-1]
        # annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # transfer boxes to sam-format 
        transfered_boxes = self.transfer_boxes_format(boxes, re_height, re_width)
        transfered_boxes = transfered_boxes.reshape(transfered_boxes.shape[0], 4)[:,[0,3,2,1]]
        transfered_boxes[:,1] = re_height - transfered_boxes[:,1]
        transfered_boxes[:,3] = re_height - transfered_boxes[:,3]
        
        return np.concatenate([transfered_boxes, confidences[:, None].numpy()], axis=1)
    
    @classmethod
    def prepare(cls):
        working_dir = get_root_folder() / "pretrained_models" / "grounding_dino"
        working_dir.mkdir(parents=True, exist_ok=True)
        N.util.download_file("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", str(working_dir))
