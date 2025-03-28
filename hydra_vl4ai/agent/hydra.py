import tensorneko_util as N
import torchvision.ops.boxes as bops
import websockets
from websockets import WebSocketClientProtocol

from .controller import ControllerLLM, ControllerDQN
from .planner import Planner
from .reasoner import Reasoner
from .smb import StateMemoryBank
from .summarizer import Summarizer
from ..evaluation.vqa_eval import GQAeval
from ..util.config import Config
from ..util.console import logger
from ..util.misc import get_hydra_root_folder


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
        self._debug = Config.debug
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
            t = timer.time(timer_msg := "Reasoner in Step 0")
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
                        t = timer.time(timer_msg := "Summarizer in Step Final")
                        logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
                        return final_result


class HydraWithRL(Hydra):

    def __init__(self, training: bool = False):
        super().__init__()
        # load config
        self.dataset = Config.base_config["dataset"]
        prompt_type = Config.base_config["prompt"]
        self.max_iterations = Config.base_config["max_iterations"]
        self._debug = Config.base_config["debug"]
        self.task = Config.base_config["task"]
        num_actions = Config.base_config["num_actions"]
        reasoner_max_retry = Config.base_config["reasoner_max_retry"]

        # load prompts
        prompt_path = get_hydra_root_folder() / "agent" / "prompt" / prompt_type
        if not prompt_path.exists():
            raise NotImplementedError(
                f"Prompt for {prompt_type} on {self.dataset} is not implemented in {prompt_path}.")

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

        embedding_prompt_base = N.io.read.text(str(prompt_path / "embedding.prompt"))
        self.controller = ControllerDQN(
            embedding_prompt_base,
            task_description_for_instruction,
            instruction_examples,
            training
        )

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

        self.evaluator = GQAeval()

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
                instruction, response_emb, selected_idx = await self.controller(query, current_step_index, instructions,
                    probs, state_memory_bank)  # TODO:MODIFY
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if instruction == "REJECT":  # TODO:MODIFY
                    continue
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
            t = timer.time(timer_msg := "Reasoner in Step 0")
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
                instruction, response_emb, selected_idx = await self.controller(query, current_step_index, instructions,
                    probs, state_memory_bank)  # TODO:MODIFY
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if instruction == "REJECT":  # TODO:MODIFY
                    continue
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
                        t = timer.time(timer_msg := "Summarizer in Step Final")
                        logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
                        return final_result

    async def train_step(self, image: bytes, query: str, ground_true=None) -> str:
        state_memory_bank = StateMemoryBank()
        async with websockets.connect(f"ws://localhost:{Config.base_config['executor_port']}/ws/") as websocket:
            match self.task:
                case "grounding":
                    return await self._train_step_grounding(image, query, state_memory_bank, websocket, ground_true)

                case "vqa":
                    return await self._train_step_vqa(image, query, state_memory_bank, websocket, ground_true)

    async def _train_step_grounding(self, image: bytes, query: str, state_memory_bank: StateMemoryBank,
        websocket: WebSocketClientProtocol, ground_true=None
    ) -> str:
        with N.util.Timer(verbose=False) as timer:
            sub_reward = 10
            pre_obs_emb = None
            for current_step_index in range(1, self.max_iterations + 1):
                # ----------------- Planner -----------------
                instructions, probs = await self.planner(query, current_step_index, state_memory_bank)
                t = timer.time(timer_msg := f"Planner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if not instructions:
                    break

                # ----------------- Controller -----------------
                instruction, response_emb, selected_idx = await self.controller(query, current_step_index, instructions,
                    probs, state_memory_bank)
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if instruction == "REJECT":
                    continue
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

                else:
                    sub_reward -= 100

                if result.type == "final":
                    sub_reward += float(bops.box_iou(result.final_result, ground_true)[0][0]) * 100

                # Calculate reward
                sub_reward -= current_step_index
                self.controller.save_model_obs_num += 1

                # update buffer and model
                if current_step_index > 1:
                    # store transition
                    self.controller.replay_buffer.push(pre_obs_emb, selected_idx, sub_reward, response_emb, done=False)
                    self.controller.reward_window.append(sub_reward)
                    self.controller.obs_no += 1
                # update model each step when buffer size bigger than batch_size.
                if len(self.controller.replay_buffer) > self.controller.batch_size:
                    for i in range(self.controller.update_times):
                        self.controller.rl_agent_model.update(replay_buffer=self.controller.replay_buffer,
                            batch_size=self.controller.batch_size)

                pre_obs_emb = response_emb  # reserve current emb as previous emb
                self.controller.dqn_explore_threshold = \
                    self.controller.dqn_explore_epsilon - self.controller.dqn_explore_epsilon_decay_rate \
                    * (self.controller.obs_no / self.controller.dqn_explore_epsilon_decay_interval)

                match result.type:
                    case "error":
                        return None
                    case "final":
                        return result.final_result
                    case "continue":
                        continue

    async def _train_step_vqa(self, image: bytes, query: str, state_memory_bank: StateMemoryBank,
        websocket: WebSocketClientProtocol, ground_true=None
    ) -> str:
        with (N.util.Timer(verbose=False) as timer):
            # initial perception
            result, code = await self.reasoner.initial_run(image, query, websocket)
            t = timer.time(timer_msg := "Reasoner in Step 0")
            logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
            if result.type == "error":
                return None
            state_memory_bank.extend_memory(
                result.feedbacks, [], [], result.variables, result.variable_names
            )

            guesses = []

            # for training
            sub_reward = 0
            pre_obs_emb = None

            for current_step_index in range(1, self.max_iterations + 1):
                # ----------------- Planner -----------------
                instructions, probs = await self.planner(query, current_step_index, state_memory_bank)
                t = timer.time(timer_msg := f"Planner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if not instructions:
                    break

                # ----------------- Controller -----------------
                instruction, response_emb, selected_idx = await self.controller(query, current_step_index, instructions,
                    probs, state_memory_bank)
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if instruction == "REJECT":
                    continue
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

                else:
                    sub_reward -= 100

                if result.type == "final":
                    if 'okvqa' in self.dataset:
                        sub_reward += self.evaluator.accuracy_one_set([result.final_result], [ground_true])
                    else:
                        sub_reward += self.evaluator.accuracy_one_one([result.final_result], [ground_true])

                # Calculate reward
                sub_reward -= current_step_index
                self.controller.save_model_obs_num += 1

                # update buffer and model
                if current_step_index > 1:
                    # store transition
                    self.controller.replay_buffer.push(pre_obs_emb, selected_idx, sub_reward, response_emb, done=False)
                    self.controller.reward_window.append(sub_reward)
                    self.controller.obs_no += 1
                # update model each step when buffer size bigger than batch_size.
                if len(self.controller.replay_buffer) > self.controller.batch_size:
                    for i in range(self.controller.update_times):
                        self.controller.rl_agent_model.update(replay_buffer=self.controller.replay_buffer,
                            batch_size=self.controller.batch_size)

                pre_obs_emb = response_emb
                self.controller.dqn_explore_threshold = \
                    self.controller.dqn_explore_epsilon - self.controller.dqn_explore_epsilon_decay_rate \
                    * (self.controller.obs_no / self.controller.dqn_explore_epsilon_decay_interval)

                match result.type:
                    case "continue":
                        continue
                    case "error":
                        return None
                    case "final":
                        # ------------------ Summarizer ------------------
                        final_result = await self.summarizer.final_guess(query, guesses)
                        t = timer.time(timer_msg := "Summarizer in Step Final")
                        logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")
                        return final_result
