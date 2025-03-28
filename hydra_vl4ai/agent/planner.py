import json

import numpy as np

from .llm import llm
from .smb.state_memory_bank import StateMemoryBank
from ..util.config import Config
from ..util.console import logger


class Planner:

    def __init__(self, 
        instruction_prompt_base: str,
        task_description_for_instruction: str,
        instruction_example: str,
        num_actions: int,
        num_trials: int
    ) -> None:
        self.instruction_prompt_base = instruction_prompt_base
        self.task_description_for_instruction = task_description_for_instruction
        self.instruction_example = instruction_example
        self.num_actions = num_actions
        self.num_trials = num_trials

    async def __call__(self, query: str, current_step_index: int, state_memory_bank: StateMemoryBank):
        prompt = self.build_prompt(query, current_step_index, state_memory_bank)
        if Config.debug:
            with open("planner.txt", "w") as f:
                f.write(prompt)
        instructions = []
        probs = []
        for _ in range(self.num_trials):
            response = await llm(Config.base_config["llm_model"], prompt) or ""
            logger.debug(f"Response from Planner: {response}")
            response = response.replace("```json", "").replace("```", "")
            # find the first "[" and the last "]" to get the json string
            response = response[response.find("["):response.rfind("]") + 1]
            instructions_match, probs_match = self.convert_chatgpt_output_to_pair(response)
            if len(instructions_match) != len(probs_match):
                continue

            instructions.extend(instructions_match)
            probs.extend(probs_match)
            logger.debug(f"Instruction from Planner: {instructions}")
            logger.debug(f"Probability from Planner: {probs}")

            if len(instructions) >= self.num_actions:
                break
        return instructions, np.array(probs)
        
    def build_prompt(self, query: str, current_step_index: int, state_memory_bank: StateMemoryBank):
        """Getting prompt based on template"""
        # prompt-for-each-query
        prompt = self.instruction_prompt_base.replace('[INSERT_QUERY_HERE]', query) # query insert
        prompt = prompt.replace('[INSERT_CURRENT_STEP_NO]', str(current_step_index)) # step number insert

        # prompt-for-query-type-about-the-dataset
        prompt = prompt.replace('[INSERT_QUERY_TYPE_HERE]', self.task_description_for_instruction) # query type
        prompt = prompt.replace('[EXAMPLE_HERE]', self.instruction_example) # query type demo/ exps

        # previous instruction
        prompt = prompt.replace('[NEED_TO_PROVIDE_PREVIOUS_INSTRUCTION]', state_memory_bank.instructions_prompt) # previous code insert
        
        # previous executed code
        prompt = prompt.replace('[MORE_CODE_WAITING]', state_memory_bank.codes_prompt) # previous code insert
        prompt = prompt.replace('[CURRENTLY_RESULT_WAITING]', state_memory_bank.feedbacks_prompt) # result description insert

        # variable details
        prompt = prompt.replace('[VARIABLE_AND_DETAILS]', state_memory_bank.variables_prompt)
        prompt = prompt.replace('[NUMBER_OF_OPTIONS]', str(self.num_actions)) # insert num__actions

        return prompt

    @staticmethod
    def convert_chatgpt_output_to_pair(input_text: str):
        """Input gpt output: str()
        Output: 
            instruction_match: list of alternative execution code
            probability_match: list of valid value of each instruction"""
        
        try:
            input_text = json.loads(input_text)
        except json.JSONDecodeError:
            logger.debug("Error: Invalid JSON format")
            return [], []
        else:
            if not isinstance(input_text, list):
                logger.debug("Error: Invalid JSON format")
                return [], []
        
        instruction_match = []
        probability_match = []
        for item in input_text:
            if 'instruction' in item and 'probability' in item:
                instruction_match.append(item['instruction'])
                probability_match.append(item['probability'])

        return instruction_match, np.array(probability_match)