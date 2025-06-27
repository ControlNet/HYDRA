import torch
from torch.nn import functional as F
from torchvision import transforms
from typing import Union

from ..util.misc import get_root_folder
from ._base import BaseModel, module_registry


@module_registry.register("clip")
class CLIPModel(BaseModel):

    def __init__(self, gpu_number=0, version="ViT-L/14@336px"):  # @336px
        super().__init__(gpu_number)

        import clip
        self.clip = clip

        model, preprocess = clip.load(version, device=self.dev,
            download_root=get_root_folder() / 'pretrained_models/clip')
        model.eval()
        model.requires_grad_ = False
        self.model = model
        self.negative_text_features = None
        self.transform = self.get_clip_transforms_from_tensor(336 if "336" in version else 224)

    # @staticmethod
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    # @staticmethod
    def get_clip_transforms_from_tensor(self, n_px=336):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            self._convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def binary_score(self, image: torch.Tensor, prompt, negative_categories=None):
        is_video = isinstance(image, torch.Tensor) and image.ndim == 4
        if is_video:  # video
            image = torch.stack([self.transform(image[i]) for i in range(image.shape[0])], dim=0)
        else:
            image = self.transform(image).unsqueeze(0).to(self.dev)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        if negative_categories is None:
            if self.negative_text_features is None:
                self.negative_text_features = self.clip_negatives(prompt_prefix)
            negative_text_features = self.negative_text_features
        else:
            negative_text_features = self.clip_negatives(prompt_prefix, negative_categories)

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(image.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        pos_text_features = self.model.encode_text(text)
        pos_text_features = F.normalize(pos_text_features, dim=-1)

        text_features = torch.concat([pos_text_features, negative_text_features], axis=0)

        # run competition where we do a binary classification
        # between the positive and all the negatives, then take the mean
        sim = (100.0 * image_features @ text_features.T).squeeze(dim=0)
        if is_video:
            query = sim[..., 0].unsqueeze(-1).broadcast_to(sim.shape[0], sim.shape[-1] - 1)
            others = sim[..., 1:]
            res = F.softmax(torch.stack([query, others], dim=-1), dim=-1)[..., 0].mean(-1)
        else:
            res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
            sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    @torch.no_grad()
    def clip_negatives(self, prompt_prefix, negative_categories=None):
        if negative_categories is None:
            with open('useful_lists/random_negatives.txt') as f:
                negative_categories = [x.strip() for x in f.read().split()]
        # negative_categories = negative_categories[:1000]
        # negative_categories = ["a cat", "a lamp"]
        negative_categories = [prompt_prefix + x for x in negative_categories]
        negative_tokens = self.clip.tokenize(negative_categories).to(self.dev)

        negative_text_features = self.model.encode_text(negative_tokens)
        negative_text_features = F.normalize(negative_text_features, dim=-1)

        return negative_text_features

    @torch.no_grad()
    def classify(self, image: Union[torch.Tensor, list], categories: list[str], return_index=True):
        is_list = isinstance(image, list)
        if is_list:
            assert len(image) == len(categories)
            image = [self.transform(x).unsqueeze(0) for x in image]
            image_clip = torch.cat(image, dim=0).to(self.dev)
        elif len(image.shape) == 3:
            image_clip = self.transform(image).to(self.dev).unsqueeze(0)
        else:  # Video (process images separately)
            image_clip = torch.stack([self.transform(x) for x in image], dim=0).to(self.dev)

        # if len(image_clip.shape) == 3:
        #     image_clip = image_clip.unsqueeze(0)

        prompt_prefix = "photo of "
        categories = [prompt_prefix + x for x in categories]
        categories = self.clip.tokenize(categories).to(self.dev)

        text_features = self.model.encode_text(categories)
        text_features = F.normalize(text_features, dim=-1)

        image_features = self.model.encode_image(image_clip)
        image_features = F.normalize(image_features, dim=-1)

        if image_clip.shape[0] == 1:
            # get category from image
            softmax_arg = image_features @ text_features.T  # 1 x n
        else:
            if is_list:
                # get highest category-image match with n images and n corresponding categories
                softmax_arg = (image_features @ text_features.T).diag().unsqueeze(0)  # n x n -> 1 x n
            else:
                softmax_arg = (image_features @ text_features.T)

        similarity = (100.0 * softmax_arg).softmax(dim=-1).squeeze(0)
        if not return_index:
            return similarity
        else:
            result = torch.argmax(similarity, dim=-1)
            if result.shape == ():
                result = result.item()
            return result

    @torch.no_grad()
    def compare(self, images: list[torch.Tensor], prompt, return_scores=False):
        images = [self.transform(im).unsqueeze(0).to(self.dev) for im in images]
        images = torch.cat(images, dim=0)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(images.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        sim = (image_features @ text_features.T).squeeze(dim=-1)  # Only one text, so squeeze

        if return_scores:
            return sim
        res = sim.argmax()
        return res

    @torch.no_grad()
    def score(self, image: torch.Tensor, prompts: list[str]):
        if isinstance(prompts, str):
            prompts = [prompts]
        image = self.transform(image).unsqueeze(0).to(self.dev)
        prompt_prefix = "photo of "
        prompts = [prompt_prefix + x for x in prompts]
        text = self.clip.tokenize(prompts).to(self.dev)
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)
        sim = (image_features @ text_features.T).squeeze(dim=0)
        return sim

    def forward(self, image, prompt, task='score', return_index=True, negative_categories=None, return_scores=False):
        if task == 'classify':
            categories = prompt
            clip_sim = self.classify(image, categories, return_index=return_index)
            out = clip_sim
        elif task == 'binary_score':
            clip_score = self.binary_score(image, prompt, negative_categories=negative_categories)
            out = clip_score
        elif task == 'score':
            clip_score = self.score(image, prompt)
            out = clip_score
        else:  # task == 'compare'
            idx = self.compare(image, prompt, return_scores)
            out = idx
        if not isinstance(out, int):
            out = out.cpu()
        return out

    @classmethod
    def prepare(cls, version="ViT-L/14@336px"):
        from clip.clip import _download, _MODELS
        _download(_MODELS[version], root=get_root_folder() / 'pretrained_models/clip')
