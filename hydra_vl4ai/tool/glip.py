import os
from typing import Union

from ..util.config import Config
from ._base import BaseModel, module_registry
import torch
from ..util.misc import get_root_folder
import tensorneko_util as N
import numpy as np
np.int = np.int_
np.float = np.float_

from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
    create_positive_map_label_to_token_from_positive_map

@module_registry.register("glip")
class GLIPModel(BaseModel):

    def __init__(self, model_size='large', gpu_number=0, *args):
        BaseModel.__init__(self, gpu_number)


        working_dir = get_root_folder() / "pretrained_models" / "GLIP"
        if model_size == 'tiny':
            config_file = working_dir / "configs/glip_Swin_T_O365_GoldG.yaml"
            weight_file = str(working_dir / "checkpoints/glip_tiny_model_o365_goldg_cc_sbu.pth")
        else:  # large
            config_file = working_dir / "configs/glip_Swin_L.yaml"
            weight_file = str(working_dir / "checkpoints/glip_large_model.pth")

        if not os.path.exists(weight_file):
            self.prepare(model_size)

        class OurGLIPDemo(GLIPDemo):

            def __init__(self, dev, *args_demo):
                detect_thresholds_glip = Config.base_config["glip_threshold"]

                kwargs = {
                    'min_image_size': 800,
                    'confidence_threshold': detect_thresholds_glip,
                    'show_mask_heatmaps': False
                }        

                self.dev = dev

                from maskrcnn_benchmark.config import cfg

                # manual override some options
                cfg.local_rank = 0
                cfg.num_gpus = 1
                cfg.merge_from_file(config_file)
                cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
                cfg.merge_from_list(["MODEL.DEVICE", self.dev])

                from transformers.utils import logging
                logging.set_verbosity_error()
                GLIPDemo.__init__(self, cfg, *args_demo, **kwargs)
                if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                    plus = 1
                else:
                    plus = 0
                self.plus = plus
                self.color = 255

            @torch.no_grad()
            def compute_prediction(self, original_image, original_caption, custom_entity=None):
                image = self.transforms(original_image)
                # image = [image, image.permute(0, 2, 1)]
                image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
                image_list = image_list.to(self.dev)
                # caption
                if isinstance(original_caption, list):

                    if len(original_caption) > 40:
                        all_predictions = None
                        for loop_num, i in enumerate(range(0, len(original_caption), 40)):
                            list_step = original_caption[i:i + 40]
                            prediction_step = self.compute_prediction(original_image, list_step, custom_entity=None)
                            if all_predictions is None:
                                all_predictions = prediction_step
                            else:
                                # Aggregate predictions
                                all_predictions.bbox = torch.cat((all_predictions.bbox, prediction_step.bbox), dim=0)
                                for k in all_predictions.extra_fields:
                                    all_predictions.extra_fields[k] = \
                                        torch.cat((all_predictions.extra_fields[k],
                                                   prediction_step.extra_fields[k] + loop_num), dim=0)
                        return all_predictions

                    # we directly provided a list of category names
                    caption_string = ""
                    tokens_positive = []
                    seperation_tokens = " . "
                    for word in original_caption:
                        tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                        caption_string += word
                        caption_string += seperation_tokens

                    tokenized = self.tokenizer([caption_string], return_tensors="pt")
                    # tokens_positive = [tokens_positive]  # This was wrong
                    tokens_positive = [[v] for v in tokens_positive]

                    original_caption = caption_string
                    # print(tokens_positive)
                else:
                    tokenized = self.tokenizer([original_caption], return_tensors="pt")
                    if custom_entity is None:
                        tokens_positive = self.run_ner(original_caption)
                    # print(tokens_positive)
                # process positive map
                positive_map = create_positive_map(tokenized, tokens_positive)

                positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map,
                                                                                                   plus=self.plus)
                self.positive_map_label_to_token = positive_map_label_to_token

                # compute predictions
                predictions = self.model(image_list, captions=[original_caption],
                                            positive_map=positive_map_label_to_token)
                predictions = [o.to(self.cpu_device) for o in predictions]
                # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

                # always single image is passed at a time
                prediction = predictions[0]

                # reshape prediction (a BoxList) into the original image size
                height, width = original_image.shape[-2:]
                # if self.tensor_inputs:
                # else:
                #     height, width = original_image.shape[:-1]
                prediction = prediction.resize((width, height))

                if prediction.has_field("mask"):
                    # if we have masks, paste the masks in the right position
                    # in the image, as defined by the bounding boxes
                    masks = prediction.get_field("mask")
                    # always single image is passed at a time
                    masks = self.masker([masks], [prediction])[0]
                    prediction.add_field("mask", masks)

                return prediction

            @staticmethod
            def to_left_right_upper_lower(bboxes):
                return [(bbox[1], bbox[3], bbox[0], bbox[2]) for bbox in bboxes]

            @staticmethod
            def to_xmin_ymin_xmax_ymax(bboxes):
                # invert the previous method
                return [(bbox[2], bbox[0], bbox[3], bbox[1]) for bbox in bboxes]

            @staticmethod
            def prepare_image(image):
                image = image[[2, 1, 0]]  # convert to bgr for opencv-format for glip
                return image

            @torch.no_grad()
            def forward(self, image: torch.Tensor, obj: Union[str, list], confidence_threshold=None, 
                        return_labels: bool = False):

                if confidence_threshold is not None:
                    original_confidence_threshold = self.confidence_threshold
                    self.confidence_threshold = confidence_threshold

                # if isinstance(object, list):
                #     object = ' . '.join(object) + ' .' # add separation tokens
                image = self.prepare_image(image)

                # Avoid the resizing creating a huge image in a pathological case
                ratio = image.shape[1] / image.shape[2]
                ratio = max(ratio, 1 / ratio)
                original_min_image_size = self.min_image_size
                if ratio > 10:
                    self.min_image_size = int(original_min_image_size * 10 / ratio)
                    self.transforms = self.build_transform()

                with torch.cuda.device(self.dev):
                    inference_output, score_output = self.inference(image, obj)

                bboxes = inference_output.bbox.cpu().numpy().astype(int)
                score_output = score_output.cpu().numpy()

                bboxes = np.concatenate([bboxes, score_output[:, None]], axis=1)
                
                # bboxes = self.to_left_right_upper_lower(bboxes)

                if ratio > 10:
                    self.min_image_size = original_min_image_size
                    self.transforms = self.build_transform()

                bboxes = torch.tensor(bboxes)

                # Convert to [left, lower, right, upper] instead of [left, upper, right, lower]
                height = image.shape[-2]
                bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1], bboxes[:, 4]], dim=1)

                # Add confidence
                if confidence_threshold is not None:
                    self.confidence_threshold = original_confidence_threshold
                if return_labels:
                    # subtract 1 because it's 1-indexed for some reason
                    return bboxes, inference_output.get_field("labels").cpu().numpy() - 1
                return bboxes

        self.glip_demo = OurGLIPDemo(*args, dev=self.dev)

    def forward(self, *args, **kwargs):
        return self.glip_demo.forward(*args, **kwargs)

    @classmethod
    def prepare(cls, model_size="large"):
        working_dir = get_root_folder() / "pretrained_models" / "GLIP"
        (working_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (working_dir / "configs").mkdir(parents=True, exist_ok=True)
        if model_size == 'tiny':
            N.util.download_file("https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth", str(working_dir / "checkpoints"))
            N.util.download_file("https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_T_O365_GoldG.yaml", str(working_dir / "configs"))
        else:  # large
            N.util.download_file("https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth", str(working_dir / "checkpoints"))
            N.util.download_file("https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml", str(working_dir / "configs"))
