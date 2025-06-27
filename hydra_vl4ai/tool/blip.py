import os
import torch
import re

from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig

from ._base import BaseModel, module_registry
from ..util.misc import get_root_folder

@module_registry.register("blip2")
class BLIP2Model(BaseModel):
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2  # The queue has additionally the time it is executing the previous forward pass

    def __init__(self, gpu_number=0, blip_v2_model_type="blip2-flan-t5-xxl"):
        super().__init__(gpu_number)

        # from lavis.models import load_model_and_preprocess
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        # https://huggingface.co/models?sort=downloads&search=Salesforce%2Fblip2-
        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']

        max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}

        if not os.path.exists(get_root_folder() / "pretrained_models" / "blip2" / blip_v2_model_type):
            self.prepare()

        self.processor = Blip2Processor.from_pretrained(get_root_folder() / "pretrained_models" / "blip2" / blip_v2_model_type)
        # Device_map must be sequential for manual GPU selection
        try:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                get_root_folder() / "pretrained_models" / "blip2" / blip_v2_model_type,
                torch_dtype="auto",
                device_map="sequential", max_memory=max_memory,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            )
        except Exception as e:
            # Clarify error message. The problem is that it tries to load part of the model to disk.
            if "had weights offloaded to the disk" in e.args[0]:
                extra_text = ' You may want to consider setting quantization to True.'
                raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
            else:
                raise e

        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.max_words = 50

    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.dev)
        generated_ids = self.model.generate(**inputs, length_penalty=1., num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = [cap.strip() for cap in
                          self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        return generated_text
    
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]

        if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
            image = [im * 255 for im in image]

        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []

        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))

        if not self.to_batch:
            response = response[0]
        return response

    @classmethod
    def prepare(cls, blip_v2_model_type: str = "blip2-flan-t5-xxl"):
        snapshot_download(f"Salesforce/{blip_v2_model_type}", local_dir=get_root_folder() / "pretrained_models" / "blip2" / blip_v2_model_type)
    