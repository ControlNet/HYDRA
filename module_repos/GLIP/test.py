from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
# Use this command for evaluate the GLIP-T model

import cv2
from matplotlib import pyplot as plt

def show_img(image, opencv:bool=True):
    if opencv:
        image = image[:,:,::-1]
    plt.imshow(image)

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


import torch
import timeit
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, create_positive_map_label_to_token_from_positive_map

class GLIPDemoT(GLIPDemo):
    def __init__(self,
                 *args,
                 **kwargs
                # cfg,
                # confidence_threshold=0.7,
                # min_image_size=None,
                # show_mask_heatmaps=False,
                # masks_per_dim=5,
                # load_model=True,
                # tensor_inputs=False
                ):
        super().__init__(*args, **kwargs)
    def compute_prediction(self, original_image, original_caption, custom_entity = None):
        # image
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        if isinstance(original_caption, list):
            # we directly provided a list of category names
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                
                tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                caption_string += word
                caption_string += seperation_tokens
            
            tokenized = self.tokenizer([caption_string], return_tensors="pt")
            tokens_positive = [tokens_positive]

            original_caption = caption_string
            print(tokens_positive)
        else:
            tokenized = self.tokenizer([original_caption], return_tensors="pt")
            if custom_entity is None:
                tokens_positive = self.run_ner(original_caption)
            print(tokens_positive)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

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
    def get_bboxes(self, image, object: str):
        image = image[[2,1,0]] # convert to bgr for opencv-format for glip
        return self.to_left_right_upper_lower(self.inference(image, object).bbox.cpu().numpy())

def load_glip(min_image_size=800, confidence_threshold=0.7, show_mask_heatmaps=False, device='cuda', tensor_inputs=True):
    working_dir = '/proj/vondrick2/sachit/projects/code-vqa/GLIP/'
    config_file = working_dir + "configs/glip_Swin_T_O365_GoldG.yaml"
    weight_file = working_dir + "checkpoints/glip_tiny_model_o365_goldg_cc_sbu.pth"
    # manual override some options
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", device])

    glip_demo = GLIPDemoT(
        cfg,
        min_image_size=min_image_size,
        confidence_threshold=confidence_threshold,
        show_mask_heatmaps=show_mask_heatmaps
    )
    glip_demo.color = 255
    return glip_demo



def glip_get_bboxes(image, object: str):
    return glip_demo.inference(image, object).bbox

def glip_predict_image(image, object: str):
    result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], object, 0.5)
    result = result[:, :, [2, 1, 0]]
    return result

# from helpers import *
import numpy as np
import torch
from torchvision import transforms as T

glip = load_glip(tensor_inputs=False)
# load with opencv
image_filepath = './images/signman4.png'

# import cv2
# imc = cv2.imread(image_filepath)
# imcpil = T.ToPILImage()(imc)
# imcpil_resize = T.Resize(300,interpolation=T.InterpolationMode.BICUBIC)(imcpil)
# imcpil_resize_tensor = T.ToTensor()(imcpil_resize)
# to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
# imcpil_resize_tensor_bgrt = to_bgr_transform(imcpil_resize_tensor)

# from PIL import Image
# imp = Image.open(image_filepath).convert("RGB")
# imp_tensor = T.ToTensor()(imp)
# imp_tensor_resize = T.Resize(300, interpolation=T.InterpolationMode.BICUBIC, antialias=True)(imp_tensor)

# imp_resize = T.Resize(300)(imp)
# imp_resize_tensor = T.ToTensor()(imp_resize)

# if cfg.INPUT.TO_BGR255:
#     to_bgr_transform = T.Lambda(lambda x: x * 255)
# else:
#     to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

# normalize_transform = T.Normalize(
#     mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
# )
# # convert numpy dtype
# transform = T.Compose(
#     [
#         T.ToPILImage(),
#         T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
#         T.ToTensor(),
#         to_bgr_transform,
#         normalize_transform,
#     ]
# )
# return transform

# ds = ImageFolder('./images', transform=transforms.ToTensor())
# impyt = ds[0][0]
# index = 2
# image_path = os.path.expanduser(os.path.join(dataset.data_path, "images", f"{dataset.df.imageId.iloc[index]}.jpg"))
# imc = cv2.imread(image_path)
# show_single_image(imc, bgr_image=True)
# show_single_image(glip.run_on_web_image(imc, 'horse')[0], bgr_image=True)