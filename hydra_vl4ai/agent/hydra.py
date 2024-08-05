import tensorneko_util as N
import websockets
from websockets import WebSocketClientProtocol

from .reasoner import Reasoner
from .smb import StateMemoryBank
from .controller import ControllerLLM
from .planner import Planner
from .summarizer import Summarizer
from ..util.console import logger
from ..util.misc import get_hydra_root_folder
from ..util.config import Config


class Hydra:
    # a base class for both HydraRL and HydraNoRL
    pass


class HydraNoRL(Hydra):

    def __init__(self):
        super().__init__()
        # load config
        dataset = Config.base_config["dataset"]
        prompt_type = Config.base_config["prompt"]
        self.max_iterations = Config.base_config["max_iterations"]
        self._debug = Config.base_config["debug"]
        self.task = Config.base_config["task"]
        num_actions = Config.base_config["num_actions"]
        reasoner_max_retry = Config.base_config["reasoner_max_retry"]

        # load prompts
        prompt_path = get_hydra_root_folder() / "agent" / "prompt" / prompt_type
        if not prompt_path.exists():
            raise NotImplementedError(f"Prompt for {prompt_type} on {dataset} is not implemented in {prompt_path}.")

        instruction_prompt_base = N.io.read.text(str(prompt_path / "instruction.prompt"))
        task_description_for_instruction = N.io.read.text(str(prompt_path / "task_description_for_instruction.prompt"))
        instruction_examples = N.io.read.text(str(prompt_path / "instruction_example.prompt"))

        code_prompt_base = N.io.read.text(str(prompt_path / "code.prompt"))
        task_description_for_code = N.io.read.text(str(prompt_path / "task_description_for_code.prompt"))
        code_example = N.io.read.text(str(prompt_path / "code_example.prompt"))

        self.planner = Planner(
            instruction_prompt_base,
            task_description_for_instruction,
            instruction_examples,
            num_actions,
            Config.base_config["planner_max_retry"])

        self.controller = ControllerLLM()

        self.reasoner = Reasoner(
            code_prompt_base,
            task_description_for_code,
            code_example,
            num_actions,
            reasoner_max_retry
        )

        match self.task:
            case "grounding":
                self.summarizer = None
            case "vqa":
                summarize_prompt_base = N.io.read.text(str(prompt_path / "summarize.prompt"))
                guess_answer_prompt_base = N.io.read.text(str(prompt_path / "guess_answer.prompt"))
                self.summarizer = Summarizer(
                    summarize_prompt_base,
                    guess_answer_prompt_base,
                    task_description_for_instruction
                )

    async def __call__(self, image: bytes, query: str) -> str:
        state_memory_bank = StateMemoryBank()
        async with websockets.connect(f"ws://localhost:{Config.base_config['executor_port']}/ws/") as websocket:
            match self.task:
                case "grounding":
                    return await self._call_grounding(image, query, state_memory_bank, websocket)
                case "vqa":
                    return await self._call_vqa(image, query, state_memory_bank, websocket)

    async def _call_grounding(self, image: bytes, query: str, state_memory_bank: StateMemoryBank,
        websocket: WebSocketClientProtocol
    ) -> str:
        with N.util.Timer(verbose=False) as timer:
            for current_step_index in range(1, self.max_iterations + 1):
                # ----------------- Planner -----------------
                instructions, probs = await self.planner(query, current_step_index, state_memory_bank)
                t = timer.time(timer_msg := f"Planner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if not instructions:
                    break

                # ----------------- Controller -----------------
                instruction = self.controller(instructions, probs)
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                # ------------------ Reasoner ------------------
                result, code = await self.reasoner(image, query, instruction, current_step_index, state_memory_bank,
                    websocket)
                t = timer.time(timer_msg := f"Reasoner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if result.type != "error":
                    assert type(code) is str
                    state_memory_bank.extend_memory(
                        result.feedbacks, [code], [instruction], result.variables, result.variable_names
                    )

                match result.type:
                    case "error":
                        return None
                    case "final":
                        return result.final_result
                    case "continue":
                        continue

    async def _call_vqa(self, image: bytes, query: str, state_memory_bank: StateMemoryBank,
        websocket: WebSocketClientProtocol
    ) -> str:
        with N.util.Timer(verbose=False) as timer:
            # initial perception
            result, code = await self.reasoner.initial_run(image, query, websocket)
            t = timer.time(timer_msg := f"Reasoner in Step 0")
            logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
            if result.type == "error":
                return None
            state_memory_bank.extend_memory(
                result.feedbacks, [], [], result.variables, result.variable_names
            )

            guesses = []

            for current_step_index in range(1, self.max_iterations + 1):
                # ----------------- Planner -----------------
                instructions, probs = await self.planner(query, current_step_index, state_memory_bank)
                t = timer.time(timer_msg := f"Planner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if not instructions:
                    break

                # ----------------- Controller -----------------
                instruction = self.controller(instructions, probs)
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                # ------------------ Reasoner ------------------
                result, code = await self.reasoner(image, query, instruction, current_step_index, state_memory_bank,
                    websocket)
                t = timer.time(timer_msg := f"Reasoner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if result.type != "error":
                    assert type(code) is str
                    state_memory_bank.extend_memory(
                        result.feedbacks, [code], [instruction], result.variables, result.variable_names
                    )
                    # ------------------ Summarizer ------------------
                    guesses.append(await self.summarizer(query, state_memory_bank))
                    t = timer.time(timer_msg := f"Summarizer in Step {current_step_index}")
                    logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                match result.type:
                    case "continue":
                        continue
                    case "error":
                        return None
                    case "final":
                        # ------------------ Summarizer ------------------
                        final_result = await self.summarizer.final_guess(query, guesses)
                        t = timer.time(timer_msg := f"Summarizer in Step Final")
                        logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
                        return final_result
