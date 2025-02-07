import abc
import pickle
import random
from collections import deque

import numpy as np
import torch

from hydra_vl4ai.util.console import logger
from .llm import llm_embedding
from .rl_dqn import DQN_EmbeddingViaLLM, ReplayBuffer
from .smb.state_memory_bank import StateMemoryBank
from ..util.config import Config
from ..util.misc import get_hydra_root_folder


class Controller(abc.ABC):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> str:
        pass


class ControllerLLM(Controller):
    """
    This is the function for not using RL controller
    but directly use the LLM score to return optimal instruction
    """

    def __call__(self, instructions: list[str], probs: np.ndarray) -> str:
        return instructions[np.argmax(probs)]


class ControllerDQN(Controller):

    def __init__(self,
        embedding_prompt_base: str,
        task_description_for_instruction: str,
        instruction_example: str,
        training: bool = False
    ):
        super().__init__()
        self.instruction_example = instruction_example
        self.embedding_prompt_base = embedding_prompt_base
        self.model_name = Config.dqn_config["model_name"]
        self.model_save_path = get_hydra_root_folder().parent / "ckpt" / self.model_name
        self.task_description_for_instruction = task_description_for_instruction
        self.training = training

        self.rl_agent_model = DQN_EmbeddingViaLLM(
            device=torch.device('cuda:0'),
            llm_embedding_dim_concat=Config.dqn_config["llm_embedding_dim"],
            mlp_hidden_dim=Config.dqn_config["mlp_hidden_dim"],
            action_dim=Config.base_config["num_actions"] + 1,
            critic_layer_num=Config.dqn_config["critic_layer_num"],
            critic_lr=float(Config.dqn_config["critic_lr"])
        )
        # load model
        self.model_full_path = self.model_save_path / "critic.pt"
        self.buffer_path = self.model_save_path / "buffer.pickle"
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        if self.model_full_path.exists():
            self.rl_agent_model.load_model(str(self.model_full_path))
            logger.info(f"Load Model Done from file: {str(self.model_full_path)}")
        elif not self.training:  # for inference, if no model, raise error
            raise RuntimeError(f"Model is not found: {self.model_full_path}")

        if self.training:  # for training
            self.rl_agent_model.train_mode()
            self.train_log_interval = Config.dqn_config["train_log_interval"]
            self.reward_window = deque(maxlen=self.train_log_interval)
            self.obs_no = 0
            self.batch_size = Config.dqn_config["batch_size"]
            self.update_times = Config.dqn_config["update_times"]
            self.save_interval = Config.dqn_config["save_interval"]
            self.save_model_obs_num = 0  # accumulate
            self.best_cum_reward = -100  # TODO:MODIFY
            self.best_score = 0
            self.learn_starts = Config.dqn_config["learn_starts"]
            self.dqn_explore_epsilon = Config.dqn_config["dqn_explore_epsilon"]
            self.dqn_explore_epsilon_decay_rate = Config.dqn_config["dqn_explore_epsilon_decay_rate"]
            self.dqn_explore_epsilon_decay_interval = Config.dqn_config["dqn_explore_epsilon_decay_interval"]
            self.dqn_explore_threshold = self.dqn_explore_epsilon - self.dqn_explore_epsilon_decay_rate \
                                         * (self.obs_no / self.dqn_explore_epsilon_decay_interval)

            # load buffer
            if self.buffer_path.exists():
                with open(self.buffer_path, "rb") as reward_buffer_container:
                    self.replay_buffer = pickle.load(reward_buffer_container)
                    reward_buffer_container.close()
            else:
                self.replay_buffer = ReplayBuffer(capacity=Config.dqn_config["buffer_size"])

        else:
            self.rl_agent_model.eval_mode()

    def save(self):
        self.rl_agent_model.save_model(self.model_full_path)
        with open(self.buffer_path, "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load(self):
        self.rl_agent_model.load_model(self.model_full_path)
        if self.training and self.buffer_path.exists():
            with open(self.buffer_path, "rb") as f:
                self.replay_buffer = pickle.load(f)

    async def __call__(self, query: str, current_step_index: int, instructions: list[str], probs: np.ndarray,
        state_memory_bank: StateMemoryBank
    ) -> tuple[str, np.ndarray, int]:
        prompt = self.build_prompt(query, current_step_index, instructions, probs, state_memory_bank)

        # get embedding from llm
        response_emb = await llm_embedding(Config.base_config["embedding_model"], prompt)

        affordance_value_array = self.rl_agent_model.get_action(obs=response_emb)

        selected_idx = np.argmax(affordance_value_array)

        # random exploration in the beginning.
        if self.training:
            # if it is in the beginning phase, do random exploration!
            if self.obs_no <= self.learn_starts or np.random.random() <= self.dqn_explore_threshold:
                selected_idx = random.choice(range(len(affordance_value_array)))

        if selected_idx != len(instructions):
            selected_instruction = instructions[selected_idx]
        else:
            selected_instruction = "REJECT"
        return selected_instruction, response_emb, selected_idx

    def build_prompt(self, query: str, current_step_index: int, instructions: list[str], probs: np.ndarray,
        state_memory_bank: StateMemoryBank
    ):
        """Getting prompt based on template"""
        # prompt-for-each-query
        prompt = self.embedding_prompt_base.replace('[INSERT_QUERY_HERE]', query)  # query insert
        prompt = prompt.replace('[INSERT_CURRENT_STEP_NO]', str(current_step_index))  # step number insert

        # prompt-for-query-type-about-the-dataset
        prompt = prompt.replace('[INSERT_QUERY_TYPE_HERE]', self.task_description_for_instruction)  # query type
        prompt = prompt.replace('[EXAMPLE_HERE]', self.instruction_example)  # query type demo/ exps

        # previous instruction
        prompt = prompt.replace('[NEED_TO_PROVIDE_PREVIOUS_INSTRUCTION]',
            state_memory_bank.instructions_prompt)  # previous code insert

        # previous executed code
        prompt = prompt.replace('[MORE_CODE_WAITING]', state_memory_bank.codes_prompt)  # previous code insert
        prompt = prompt.replace('[CURRENTLY_RESULT_WAITING]',
            state_memory_bank.feedbacks_prompt)  # result description insert

        # variable details
        prompt = prompt.replace('[VARIABLE_AND_DETAILS]', state_memory_bank.variables_prompt)

        # current instructions/probs
        prompt = prompt.replace('[CURRENT_OPTION]', str(instructions))  # instruction options
        prompt = prompt.replace('[CURRENT_OPTION_PROBABILITY]', str(probs))  # probs of instruction options

        return prompt
