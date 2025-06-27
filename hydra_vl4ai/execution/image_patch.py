from __future__ import annotations
import asyncio

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n
import tensorneko_util as N

from .toolbox import forward
from ..agent.llm import llm
from ..agent.smb import StateMemoryBank
from ..util.misc import get_hydra_root_folder, load_json
from ..util.config import Config


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """
    possible_options = load_json(get_hydra_root_folder() / "assets" / "possible_options.json")

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
        right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
        parent_img_patch: ImagePatch = None, image_name='original_image', state_memory_bank: StateMemoryBank = None,
        confidence: float = 1.0
    ):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """
        assert state_memory_bank is not None, "StateMemoryBank is not provided"
        self.state_memory_bank = state_memory_bank
        self.confidence = confidence

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1] - upper:image.shape[1] - lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]
        self.image_name = image_name

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.ratio_box_area_to_image_area = Config.base_config['ratio_box_area_to_image_area']
        self.verify_property_model = Config.base_config["verify_property_model"]
        self.crop_larger_margin = Config.base_config['crop_larger_margin']

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_list: list[str], get_feed_back: bool = True) -> dict[str, list[ImagePatch]]:
        """Returns a dictionary. Each pair includes a list of ImagePatch objects matching the object_name contained in the crop, if any are found.
        Otherwise, returns an empty dict.
        Parameters
        ----------
        object_list : list
            A list of the name of the object to be found
        get_feed_back : bool
            A boolean value to determine whether to get feedback or not

        Returns
        -------
        dict
            Each pair includes a list of ImagePatch objects matching the object_name contained in the crop
        """
        detected_dict = {}

        for object_name in object_list:
            if object_name in ["object", "objects"]:
                all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
            else:
                if object_name == 'person':
                    all_object_coordinates = self.forward(Config.base_config["grounding_model"], self.cropped_image,
                        'people')
                else:
                    all_object_coordinates = self.forward(Config.base_config["grounding_model"], self.cropped_image,
                        object_name)

            if len(all_object_coordinates) > 0:

                threshold = self.ratio_box_area_to_image_area
                if threshold > 0:
                    area_im = self.width * self.height
                    all_areas = torch.tensor([(coord[2] - coord[0]) * (coord[3] - coord[1]) / area_im
                        for coord in all_object_coordinates])
                    mask = all_areas > threshold

                    all_object_coordinates = all_object_coordinates[mask]

                current_img_no = 0

                if get_feed_back:
                    # Adding description
                    self.state_memory_bank.find_general_add_feedback(all_object_coordinates, object_name,
                        self.image_name)

                # crop image
                detected_dict[object_name] = [
                    self.crop(*coordinates[:4], image_name=f'{object_name}_{img_no_ + 1}_in_{self.image_name}',
                        confidence=coordinates[4]
                    ) for img_no_, coordinates in enumerate(all_object_coordinates)]

                for each_obj in detected_dict[object_name]:
                    current_img_no += 1
                    bd_box_prediction = str(
                        [each_obj.left, self.original_image.shape[1] - each_obj.upper, each_obj.right,
                            self.original_image.shape[1] - each_obj.lower])  # the detected object bounding box to str

                    if get_feed_back:
                        self.state_memory_bank.find_bounding_box_add_feedback(object_name, current_img_no,
                            self.image_name, bd_box_prediction)  # add the detection result into description

            elif get_feed_back:
                self.state_memory_bank.find_cant_found_add_feedback(object_name)  # cant find this object
                detected_dict[object_name] = []

        return detected_dict

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find([object_name], get_feed_back=False)[object_name]

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)

        exist_res = len(filtered_patches) > 0

        # add description

        self.state_memory_bank.exist_add_feedback(object_name, self.image_name, exist_res)
        # execution_result_description += f'\nThe existence of {object_name} in image patch {self.image_name} is: {len(filtered_patches) > 0}'

        return exist_res

    def _score(self, category: str, negative_categories=None, model='xvlm') -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        # if model == 'clip':
        #     res = self.forward('clip', self.cropped_image, category, task='score',
        #                        negative_categories=negative_categories)
        # elif model == 'tcl':
        #     res = self.forward('tcl', self.cropped_image, category, task='score')
        # else:  # xvlm
        #     task = 'binary_score' if negative_categories is not None else 'score'
        #     res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
        #     res = res.item()
        task = 'binary_score' if negative_categories is not None else 'score'
        res = self.forward(model, self.cropped_image, category, task=task, negative_categories=negative_categories)
        res = res.item()
        return res

    def _detect(self, category: str, thresh, negative_categories=None, model='xvlm') -> bool:
        output_of_res = self._score(category, negative_categories, model) > thresh

        # get global description.
        self.state_memory_bank.verify_add_feedback(category, self.image_name, output_of_res)
        # execution_result_description += f'\nThe verification of {category} in {self.image_name} is: {output_of_res}'

        return output_of_res

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]

        if self.verify_property_model == "xvlm":
            return self._detect(name, negative_categories=negative_categories,
                thresh=Config.base_config["verify_property_thresh_xvlm"], model='xvlm')
        elif self.verify_property_model == "clip":
            return self._detect(name, negative_categories=negative_categories,
                thresh=Config.base_config["verify_property_thresh_clip"], model='clip')
        else:
            raise NotImplementedError

    def verify_property_score(self, object_name: str, attribute: str) -> float:
        name = f"{attribute} {object_name}"
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        return self._score(name, negative_categories=negative_categories, model=self.verify_property_model)

    def classify_object_attributes(self, object_name: str, attributes: list[str]) -> np.ndarray:
        """
        Returns the classification scores for the object's attributes.
        """
        return self.forward(Config.base_config["classify_object_attributes_model"], self.cropped_image,
            [f"{attribute} {object_name}" for attribute in attributes], task="score", negative_categories=None).numpy()

    def classify_object_types(self, types: list[str]) -> np.ndarray:
        """
        Returns the classification scores for the object's types.
        """
        return self.forward(Config.base_config["classify_object_type_model"], self.cropped_image, types, task="score",
            negative_categories=None).numpy()

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        image = self.cropped_image
        text = option_list_to_use
        res = self.forward(self.verify_property_model, image, text, task='score')
        res = res.argmax().item()
        selected = res

        # get global description.
        self.state_memory_bank.get_best_text_match_add_feedback(self.image_name, str(option_list),
            option_list[selected])
        # execution_result_description += f"\nThe best text match for image patch {self.image_name} among ({str(option_list)}) is: '{option_list[selected]}'"

        return option_list[selected]

    def simple_query(self, question: str = 'What is this?', qa_mode: bool = True) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        if qa_mode:
            query_answer = self.forward(Config.base_config["vlm_model"], self.cropped_image, question, task='qa')
            # get global description.
            self.state_memory_bank.get_answer_of_simple_question_add_feedback(self.image_name, question, query_answer)
        else:
            query_answer = self.forward(Config.base_config["vlm_caption_model"], self.cropped_image, question, task='caption')
            # get global description.
            self.state_memory_bank.get_caption_add_feedback(self.image_name, query_answer)

        return query_answer

    def compute_depth(self) -> float:
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1] - self.upper:original_image.shape[1] - self.lower,
        self.left:self.right]

        median_depth = depth_map.median()

        # get global description.
        self.state_memory_bank.depth_add_feedback(self.image_name, median_depth)
        # execution_result_description += f"\nThe median depth for image patch {self.image_name} is: {median_depth}"

        return median_depth  # Ideally some kind of mode, but median is good enough for now

    def crop(self, left: int | float, lower: int | float, right: int | float, upper: int | float, image_name: str,
        confidence: float = None
    ) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if self.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
            parent_img_patch=self, image_name=image_name, state_memory_bank=self.state_memory_bank,
            confidence=confidence if confidence is not None else self.confidence)

    def overlaps_with(self, other_patch: ImagePatch) -> bool:
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        left, lower, right, upper = other_patch.left, other_patch.lower, other_patch.right, other_patch.upper

        overlaps_res = self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower
        # get global description.
        self.state_memory_bank.overlapping_add_feedback(self.image_name, other_patch.image_name, overlaps_res)
        # execution_result_description += f"\nThe check result for overlapping between image patch {self.image_name} and image patch located in {[left, self.original_image.shape[1]-upper, right, self.original_image.shape[1]-lower]} is: {overlaps_res}"

        return overlaps_res

    def segment(self):
        mask = self.forward("efficient_sam", self.original_image, self.to_bbox()[:4])
        return mask

    def llm_query(self, question: str, context: str = None, long_answer: bool = True) -> str:
        return llm_query(question, context, long_answer, state_memory_bank=self.state_memory_bank)

    def print_image(self, size: tuple[int, int] = None):
        raise NotImplementedError

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)

    def to_bbox(self):
        x0 = self.left + 10
        x1 = self.right - 10
        y0 = self.original_image.shape[1] - self.upper + 10
        y1 = self.original_image.shape[1] - self.lower - 10
        return [x0, y0, x1, y1, float(self.confidence)]


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
    Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = Config.base_config["verify_property_model"]

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores

    best_image_matching_out = list_patches[scores]

    # get global description.
    list_patches[0].state_memory_bank.best_image_match_add_feedback(content, best_image_matching_out.image_name)
    # execution_result_description += f"\nThe best image match patch (most likely to contain the {str(content)}) is: {best_image_matching_out.image_name}"

    return best_image_matching_out


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()
        # get global description.
        patch_a.state_memory_bank.dist_add_feedback(patch_a.image_name, patch_b.image_name, dist)
    else:
        dist = abs(patch_a - patch_b)

    # execution_result_description += f"\nThe distance between the edges of {patch_a.image_name} and {patch_b.image_name} is: {dist}"
    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(question, context=None, long_answer=True, state_memory_bank=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    question: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if context != None:
        prompt_ = f'The context information is: {context}. '
    else:
        prompt_ = ''

    prompt_ += f'Could you help me answer the question: {question}.'

    if not long_answer:
        prompt_ += 'Please provide only a few-word answer. Be very concise, no ranges, no doubt.'
    try:
        return_answer = asyncio.run(llm(Config.base_config["llm_model"], prompt_)) or ""
    except Exception:
        return_answer = 'not answer from gpt'

    # get global description.
    state_memory_bank.llm_add_feedback(question, context, return_answer)

    return return_answer


def get_sorted_patches_left_to_right(list_patches: list[ImagePatch]) -> list[ImagePatch]:
    """
    Sorts patches according to their horizontal centers from left to right.
    """
    list_patches.sort(key=lambda patch: patch.horizontal_center)
    if len(list_patches) > 0:
        list_patches[0].state_memory_bank.get_sorted_patches_left_to_right_message_save(list_patches)
    return list_patches


def get_sorted_patches_bottom_to_top(list_patches: list[ImagePatch]) -> list[ImagePatch]:
    """
    Sorts patches according to their vertical centers from bottom/low to top/up.
    """
    list_patches.sort(key=lambda patch: patch.vertical_center)
    if len(list_patches) > 0:
        list_patches[0].state_memory_bank.get_sorted_patches_bottom_to_top_message_save(list_patches)
    return list_patches


def get_sorted_patches_front_to_back(list_patches: list[ImagePatch]) -> list[ImagePatch]:
    """
    Sorts patches according to how far from camera they are. Sorts patches from front/close to back/far.
    """
    list_patches.sort(key=lambda patch: patch.compute_depth())
    if len(list_patches) > 0:
        list_patches[0].state_memory_bank.get_sorted_patches_front_to_back_message_save(list_patches)
    return list_patches


def get_middle_patch(list_patches: list[ImagePatch]) -> ImagePatch:
    """
    Returns the middle patch from list of patches.
    """
    len_list = len(list_patches)
    middle_patch = get_sorted_patches_left_to_right(list_patches)[len_list // 2]
    if len_list > 0:
        list_patches[0].state_memory_bank.get_middle_patch_message_save(middle_patch.image_name)
    return middle_patch


def get_patch_left_of(patch: ImagePatch) -> ImagePatch:
    """
    Returns the left part of the original image for the given patch.
    """
    part_of_given = patch.parent_img_patch.crop(patch.parent_img_patch.left, patch.parent_img_patch.lower, \
        patch.horizontal_center, patch.parent_img_patch.upper, f'left_partof_{patch.image_name}')

    return part_of_given


def get_patch_right_of(patch: ImagePatch) -> ImagePatch:
    """
    Returns the right part of the original image for the given patch.
    """
    part_of_given = patch.parent_img_patch.crop(patch.horizontal_center, patch.parent_img_patch.lower, \
        patch.parent_img_patch.right, patch.parent_img_patch.upper, f'right_partof_{patch.image_name}')

    return part_of_given


def get_patch_above_of(patch: ImagePatch) -> ImagePatch:
    """
    Returns the above part of the original image for the given patch.
    """
    part_of_given = patch.parent_img_patch.crop(patch.parent_img_patch.left, patch.parent_img_patch.vertical_center, \
        patch.parent_img_patch.right, patch.parent_img_patch.upper, f'above_partof_{patch.image_name}')

    return part_of_given


def get_patch_below_of(patch: ImagePatch) -> ImagePatch:
    """
    Returns the below part of the original image for the given patch.
    """
    part_of_given = patch.parent_img_patch.crop(patch.parent_img_patch.left, patch.parent_img_patch.lower, \
        patch.parent_img_patch.right, patch.parent_img_patch.vertical_center, f'below_partof_{patch.image_name}')

    return part_of_given


def get_patch_closest_to_anchor_object(list_patches: list[ImagePatch], anchor_patch: ImagePatch) -> ImagePatch:
    """
    Returns the patch closest to the anchor patch from a list of patches.
    """
    list_patches.sort(key=lambda patch: distance(patch, anchor_patch))
    if len(list_patches) > 0:
        list_patches[0].state_memory_bank.get_patch_closest_to_anchor_object_message_save(list_patches[0].image_name,
            anchor_patch.image_name)
    return list_patches[0]


def get_patch_farthest_to_anchor_object(list_patches: list[ImagePatch], anchor_patch: ImagePatch) -> ImagePatch:
    """
    Returns the patch closest to the anchor patch from a list of patches.
    """
    list_patches.sort(key=lambda patch: distance(patch, anchor_patch))
    if len(list_patches) > 0:
        list_patches[0].state_memory_bank.get_patch_farthest_to_anchor_object_message_save(list_patches[-1].image_name,
            anchor_patch.image_name)
    return list_patches[-1]


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub(r"[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric
