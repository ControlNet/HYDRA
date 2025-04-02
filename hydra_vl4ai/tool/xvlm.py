import os
from pathlib import Path

from transformers import BertTokenizer
from torchvision import transforms
import torch.nn.functional as F
import torch
import re

from ..util.misc import get_root_folder, get_hydra_root_folder
from ..util.console import logger
from ._base import BaseModel, module_registry


@module_registry.register("xvlm")
class XVLMModel(BaseModel):
    def __init__(self, gpu_number=0,
                 path_checkpoint=get_root_folder() / 'pretrained_models/xvlm/retrieval_mscoco_checkpoint_9.pth'):
        from .model.xvlm.xvlm import XVLMBase

        super().__init__(gpu_number)

        image_res = 384
        self.max_words = 30
        config_xvlm = {
            'image_res': image_res,
            'patch_size': 32,
            'text_encoder': 'bert-base-uncased',
            'block_num': 9,
            'max_tokens': 40,
            'embed_dim': 256,
        }

        vision_config = {
            'vision_width': 1024,
            'image_res': 384,
            'window_size': 12,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32]
        }
        model = XVLMBase(config_xvlm, use_contrastive_loss=True, vision_config=vision_config)
        if not os.path.exists(path_checkpoint):
            self.prepare()
        checkpoint = torch.load(path_checkpoint, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) > 0:
            logger.info('XVLM Missing keys: ', msg.missing_keys)

        model = model.to(self.dev)
        model.eval()

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_res, image_res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        with open(get_hydra_root_folder() / 'assets/random_negatives.txt') as f:
            self.negative_categories = [x.strip() for x in f.read().split()]

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

    @torch.no_grad()
    def score(self, images, texts):

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(images, list):
            images = [images]

        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.dev)

        texts = [self.pre_caption(text, self.max_words) for text in texts]
        text_input = self.tokenizer(texts, padding='longest', return_tensors="pt").to(self.dev)

        image_embeds, image_atts = self.model.get_vision_embeds(images)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.model.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.model.get_features(image_embeds, text_embeds)
        logits = image_feat @ text_feat.t()

        return logits

    @torch.no_grad()
    def binary_score(self, image, text, negative_categories):
        # Compare with a pre-defined set of negatives
        texts = [text] + negative_categories
        sim = 100 * self.score(image, texts)[0]
        res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                   sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    def forward(self, image, text, task='score', negative_categories=None):
        if task == 'score':
            score = self.score(image, text)
        else:  # binary
            score = self.binary_score(image, text, negative_categories=negative_categories)
        return score.cpu()

    @classmethod
    def prepare(cls):
        import gdown
        (get_root_folder() / 'pretrained_models/xvlm/').mkdir(parents=True, exist_ok=True)
        if (model_path := get_root_folder() / 'pretrained_models/xvlm/retrieval_mscoco_checkpoint_9.pth').exists():
            return
        gdown.download(
            "https://drive.google.com/u/0/uc?id=1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN",
            str(model_path),
        )
