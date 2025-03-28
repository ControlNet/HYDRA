import io
import json

import websockets
from PIL import Image

from .llm import llm
from .smb.state_memory_bank import StateMemoryBank
from ..util.config import Config
from ..util.console import logger
from ..util.message import ExecutionRequest, ExecutionResult


class Reasoner:
    def __init__(self, 
        code_prompt_base: str,
        task_description_for_code: str,
        code_example: str,
        num_actions: int,
        num_trials: int,
    ) -> None:
        self.code_prompt_base = code_prompt_base
        self.dataset_description_for_code = task_description_for_code
        self.code_example = code_example
        self.num_actions = num_actions
        self.num_trials = num_trials

    async def __call__(self, 
        image_buffer: bytes | None, 
        query: str, 
        instruction: str, 
        current_step_index: int, 
        state_memory_bank: StateMemoryBank, 
        websocket: websockets.WebSocketClientProtocol
    ) -> tuple[ExecutionResult, str | None]:
        prompt = self.build_prompt(query, instruction, current_step_index, state_memory_bank)
        if Config.debug:
            with open("reasoner.txt", "w") as f:
                f.write(prompt)

        assert self.num_trials > 0
        result = ExecutionResult.from_dict({
            "type": "error",
            "feedbacks": [],
            "variables": [],
            "variable_names": [],
            "final_result": ""
        })

        for _ in range(self.num_trials):
            code = await llm(Config.base_config["llm_code_model"], prompt)
            if code is None:
                continue
            
            # clean codes
            code = code.replace("image_patch = ImagePatch(image)", "")
            code = code.replace("```python", "").replace("```", "")
            
            send_image = image_buffer is not None
            message = ExecutionRequest(code, send_image)
            await websocket.send(json.dumps(message.to_dict()))
            if send_image:
                await websocket.send(image_buffer)

            logger.debug("---------------Sending code to executor:---------------")
            logger.debug(code)

            result: ExecutionResult = json.loads(await websocket.recv(), object_hook=ExecutionResult.from_dict)

            logger.debug("---------------Received result from executor:---------------")
            logger.debug(result)

            match result.type:
                case "error":
                    continue
                case "final" | "continue":
                    return result, code
                
        return result, None  # it should be the error result

    def build_prompt(self, query: str, instruction: str, current_step_index: int, state_memory_bank: StateMemoryBank):
        """Getting prompt based on template"""
        # prompt-for-each-query
        prompt = self.code_prompt_base.replace('[INSERT_QUERY_HERE]', query) # query insert
        prompt = prompt.replace('[INSERT_CURRENT_STEP_NO]', str(current_step_index)) # step number insert

        # prompt-for-query-type-about-the-dataset
        prompt = prompt.replace('[INSERT_QUERY_TYPE_HERE]', self.dataset_description_for_code) # query type
        prompt = prompt.replace('[EXAMPLE_HERE]', self.code_example) # query type demo/ exps

        # previous instruction
        prompt = prompt.replace('[NEED_TO_PROVIDE_PREVIOUS_INSTRUCTION]', state_memory_bank.instructions_prompt) # previous code insert
        
        # previous executed code
        prompt = prompt.replace('[MORE_CODE_WAITING]', state_memory_bank.codes_prompt) # previous code insert
        prompt = prompt.replace('[CURRENTLY_RESULT_WAITING]', state_memory_bank.feedbacks_prompt) # result description insert

        # variable details
        prompt = prompt.replace('[VARIABLE_AND_DETAILS]', state_memory_bank.variables_prompt)

        prompt = prompt.replace('[INSERT_CURRENT_INSTRUCTION_HERE]', instruction) # current instruction insert

        return prompt
    
    async def initial_run(self,
        image_buffer: bytes | None,
        query: str,
        websocket: websockets.WebSocketClientProtocol
    ):
        # add some initial perception to the state memory bank (etc. the caption from BLIP)
        code = f"image_patch.simple_query(\"{query}\", qa_mode=False)"
        assert image_buffer is not None
        message = ExecutionRequest(code, True)
        await websocket.send(json.dumps(message.to_dict()))
        await websocket.send(image_buffer)

        logger.debug("---------------Sending code to executor:---------------")
        logger.debug(code)

        result: ExecutionResult = json.loads(await websocket.recv(), object_hook=ExecutionResult.from_dict)

        logger.debug("---------------Received result from executor:---------------")
        logger.debug(result)

        # load the image to check the width and height
        image = Image.open(io.BytesIO(image_buffer))
        width, height = image.size

        result.variables = [
            f"image_patch: ImagePatch(0, 0, {width}, {height})"
        ]

        result.variable_names = [
            "image_patch"
        ]

        return result, code
