import os
import torch
import re
from torchvision import transforms
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from huggingface_hub import snapshot_download

from ._base import BaseModel, module_registry
from ..util.misc import get_root_folder


@module_registry.register("llava1.5")
class LLaVA(BaseModel):

    def __init__(self, gpu_number=0, model_name: str = "liuhaotian/llava-v1.5-7b"):
        super().__init__(gpu_number)
        self.model_path = get_root_folder() / "pretrained_models" / "llava" / model_name.split("/")[-1]
        self.model_name = get_model_name_from_path(str(self.model_path))
        if not os.path.exists(self.model_path):
            self.prepare(model_name)
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path,
            model_name=self.model_name, model_base=None, load_8bit=False, load_4bit=False)

    @classmethod
    def prepare(cls, model_name: str = "liuhaotian/llava-v1.5-7b"):
        snapshot_download(model_name,
            local_dir=get_root_folder() / "pretrained_models" / "llava" / model_name.split("/")[-1])

    @torch.no_grad()
    def forward(self, input_image, query):
        # Model Constants
        IGNORE_INDEX = -100
        IMAGE_TOKEN_INDEX = -200
        DEFAULT_IMAGE_TOKEN = "<image>"
        DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
        DEFAULT_IM_START_TOKEN = "<im_start>"
        DEFAULT_IM_END_TOKEN = "<im_end>"
        IMAGE_PLACEHOLDER = "<image-placeholder>"

        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # from tensor to pil
        input_image = transforms.functional.to_pil_image(input_image).convert('RGB')
        images = [input_image]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda(self.dev)
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if 0.15 > 0 else False,
                temperature=0.15,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs
