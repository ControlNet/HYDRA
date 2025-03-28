from .llm import llm
from .smb.state_memory_bank import StateMemoryBank
from ..util.config import Config


class Summarizer:

    def __init__(self,
        summarize_prompt_base: str,
        guess_answer_prompt_base: str,
        task_description_for_instruction: str
    ) -> None:
        self.summarize_prompt_base = summarize_prompt_base
        self.guess_answer_prompt_base = guess_answer_prompt_base
        self.task_description_for_instruction = task_description_for_instruction

    async def __call__(self, query: str, state_memory_bank: StateMemoryBank) -> str:
        prompt = self.build_summarizer_prompt(query, state_memory_bank)
        if Config.debug:
            with open("summarizer.txt", "w") as f:
                f.write(prompt)
        response = await llm(Config.base_config["llm_model"], prompt) or ""
        return response

    async def final_guess(self, query: str, guesses: list[str]) -> str:
        prompt = self.build_guess_prompt(query, guesses)
        response = await llm(Config.base_config["llm_model"], prompt) or ""
        return response

    def build_summarizer_prompt(self, query: str, state_memory_bank: StateMemoryBank):
        """Getting prompt based on template"""
        # prompt-for-each-query
        prompt = self.summarize_prompt_base.replace('[INSERT_QUERY_HERE]', query)  # query insert

        # prompt-for-query-type-about-the-dataset
        prompt = prompt.replace('[INSERT_QUERY_TYPE_HERE]', self.task_description_for_instruction)  # query type

        # previous instruction
        prompt = prompt.replace('[NEED_TO_PROVIDE_PREVIOUS_INSTRUCTION]',
            state_memory_bank.instructions_prompt)  # previous code insert

        # previous executed code
        prompt = prompt.replace('[MORE_CODE_WAITING]', state_memory_bank.codes_prompt)  # previous code insert
        prompt = prompt.replace('[CURRENTLY_RESULT_WAITING]',
            state_memory_bank.feedbacks_prompt)  # result description insert

        # variable details
        prompt = prompt.replace('[VARIABLE_AND_DETAILS]', state_memory_bank.variables_prompt)

        return prompt

    def build_guess_prompt(self, query: str, guesses: list[str]):
        """Getting prompt based on template"""
        # prompt-for-each-query
        prompt = self.guess_answer_prompt_base.replace('[INSERT_QUERY_HERE]', query)  # query insert

        # prompt-for-query-type-about-the-dataset
        prompt = prompt.replace('[INSERT_QUERY_TYPE_HERE]', self.task_description_for_instruction)  # query type

        # previous instruction
        prompt = prompt.replace('[INSERT_GUESS_HERE]', str(guesses))

        return prompt
