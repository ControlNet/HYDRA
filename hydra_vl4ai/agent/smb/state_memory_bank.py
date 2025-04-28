from __future__ import annotations
import json


class StateMemoryBank:
    """The state memory bank implementation.
    In the agent, the state memory bank is used to store the state of the agent.
    In the executor, the state memory bank is used to store the state of the current iteration, and then update to the main state in agent.
    """
    feedbacks: list[str]
    codes: list[str]
    instructions: list[str]
    variables: list[str]
    variable_names: list[str]

    def __init__(self):
        self.reset()

    def reset(self):
        self.feedbacks = []
        self.codes = []
        self.instructions = []
        self.variables = []  # variable name and descriptions
        self.variable_names = []  # variable names

    def clone(self):
        new_smb = StateMemoryBank()
        new_smb.feedbacks = self.feedbacks.copy()
        new_smb.codes = self.codes.copy()
        new_smb.instructions = self.instructions.copy()
        new_smb.variables = self.variables.copy()
        new_smb.variable_names = self.variable_names.copy()
        return new_smb
    
    def to_dict(self) -> dict[str, list[str]]:
        return {
            "feedbacks": self.feedbacks,
            "codes": self.codes,
            "instructions": self.instructions,
            "variables": self.variables,
            "variable_names": self.variable_names,
        }
    
    @classmethod
    def from_json(cls, json_str: str) -> StateMemoryBank:
        data = json.loads(json_str)
        smb = cls()
        smb.feedbacks = data["feedbacks"]
        smb.codes = data["codes"]
        smb.instructions = data["instructions"]
        smb.variables = data["variables"]
        smb.variable_names = data["variable_names"]
        return smb

    @property
    def instructions_prompt(self):
        return "\n" + "\n".join(self.instructions)

    @property
    def feedbacks_prompt(self):
        return "\n" + "\n".join(self.feedbacks)

    @property
    def codes_prompt(self):
        return "\n" + "\n".join(self.codes)

    @property
    def variables_prompt(self):
        return "\n" + "\n".join(self.variables)

    def extend_memory(self,
        other_feedbacks: list[str],
        other_codes: list[str],
        other_instructions: list[str],
        other_variables: list[str],
        other_variable_names: list[str]
    ):
        self.feedbacks.extend(other_feedbacks)
        self.codes.extend(other_codes)
        self.instructions.extend(other_instructions)
        self.variables.extend(other_variables)
        self.variable_names.extend(other_variable_names)

    def find_general_add_feedback(self, all_object_coordinates, object_name, image_name):
        if len(all_object_coordinates) == 1:
            self.feedbacks.append(
                f'Detection result: Only one {object_name} has been detected in {image_name}.')
        else:
            self.feedbacks.append(
                f'Detection result: {len(all_object_coordinates)} {object_name} have been detected in {image_name}.')

    def find_bounding_box_add_feedback(self, object_name, current_img_no, image_name, bd_box_prediction):
        pass

    def find_cant_found_add_feedback(self, object_name):
        self.feedbacks.append(
            f'Detection result: no {object_name} has been detected.')

    def exist_add_feedback(self, object_name, image_name, exisit_res):
        self.feedbacks.append(
            f'The existence of {object_name} in image patch {image_name} is: {exisit_res}')

    def verify_add_feedback(self, category, image_name, output_of_res):
        self.feedbacks.append(
            f'The verification of {category} in {image_name} is: {output_of_res}')

    def get_best_text_match_add_feedback(self, image_name, option_list, selected_option):
        self.feedbacks.append(
            f"The best text match for image patch {image_name} among ({option_list}) is: '{selected_option}'")

    def get_caption_add_feedback(self, image_name, query_answer_):
        self.feedbacks.append(
            f"The caption for image patch {image_name} is: {query_answer_}")

    def get_answer_of_simple_question_add_feedback(self, image_name, question, query_answer_):
        self.feedbacks.append(
            f"The answer for image patch {image_name} in response to the question '{question}' is: {query_answer_}")

    def depth_add_feedback(self, image_name, median_depth):
        self.feedbacks.append(
            f"\nThe median depth for image patch {image_name} is: {median_depth}")

    def overlapping_add_feedback(self, image_name, other_name, overlaps_res):
        self.feedbacks.append(
            f"\nThe check result for overlapping between image patch {image_name} and image patch {other_name} is: {overlaps_res}")

    def best_image_match_add_feedback(self, content, image_name):
        self.feedbacks.append(
            f"\nThe best image match patch (most likely to contain the {str(content)}) is: {image_name}")

    def dist_add_feedback(self, image_name_a, image_name_b, dist):
        self.feedbacks.append(
            f"\nThe distance between the edges of {image_name_a} and {image_name_b} is: {dist}")

    def llm_add_feedback(self, query, context, return_answer):
        self.feedbacks.append(
            f"\nThe obtained answer from LLM to the question, {query} with the additional context of {context} is: {return_answer}")

    # check result
    def return_final_answer_have_too_many_options(self, number_imagepatch_infinal):
        self.feedbacks.append(
            f"\nThere are {number_imagepatch_infinal} ImagePatch in final_answer, you should only get one ImagePatch as the target patch.")

    def return_final_answer_should_be_ImagePatch(self):
        self.feedbacks.append(
            f"\nThe final_answer is not an ImagePatch, please ensure that only one ImagePatch is returned.")

    def return_final_answer_should_be_string(self):
        self.feedbacks.append(
            f"\nThe final_answer is not a string, please ensure that only string is returned.")

    # sorting
    def get_sorted_patches_left_to_right_message_save(self, name):
        self.feedbacks.append(
            f"\nThe patches list has been sorted from left to right (horizontal). Now, the first patch in the list corresponds to the leftest position, while the last one corresponds to the rightest position")

    def get_sorted_patches_bottom_to_top_message_save(self, name):
        self.feedbacks.append(
            f"\nThe patches list has been sorted from bottom to top (vertical). Now, the first patch in the list corresponds to the bottom/low/below position, while the last one corresponds to the top/up/above position.")

    def get_sorted_patches_front_to_back_message_save(self, name):
        self.feedbacks.append(
            f"\nThe patches list has been sorted from front/close to back/far. Now, the first patch in the list corresponds to the front/closest position, while the last one corresponds to the back/farther position.")

    # get middle
    def get_middle_patch_message_save(self, name):
        self.feedbacks.append(f"\nThe {name} is the middle one in the list.")

    # get closest
    def get_patch_closest_to_anchor_object_message_save(self, name, anchor_name):
        self.feedbacks.append(
            f"\nThe {name} is the closest one to {anchor_name}.")

    # get farthest
    def get_patch_farthest_to_anchor_object_message_save(self, name, anchor_name):
        self.feedbacks.append(
            f"\nThe {name} is the farthest one to {anchor_name}.")
