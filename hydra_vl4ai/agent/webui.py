import gradio as gr
import tensorneko_util as N
import websockets
from websockets import WebSocketClientProtocol

from .hydra import HydraNoRL
from .smb import StateMemoryBank
from ..util.config import Config
from ..util.console import logger


class HydraNoRLWeb(HydraNoRL):

    def __init__(self):
        super().__init__()

        with gr.Blocks() as self.gradio_app:
            self.output = gr.Chatbot(label="Hydra", type="messages")
            self.current_state = gr.Markdown(label="Current state")

            self.image = gr.File(label="Image to query")
            self.query = gr.Textbox(label="Query description", placeholder="Ask me anything!", submit_btn=True)

            self.query.submit(self._gradio_call, [self.image, self.query], [self.output, self.current_state])

    async def _gradio_call(self, image_path: str, query: str):
        logger.info(f"Image path: {image_path}")

        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        self.bank = StateMemoryBank()

        messages = [gr.ChatMessage('user', [image_path, query])]
        yield messages, self._format_state_memory()

        async with websockets.connect(
            f"ws://localhost:{Config.base_config['executor_port']}/ws/") as ws:  # type: ignore
            async for chunk in self._call_vqa(image_bytes, query, self.bank, ws):
                if chunk[0] == "start":
                    continue
                elif chunk[0] == "panic":
                    messages.append(gr.ChatMessage("assistant", "Sorry, please check the logs for more information."))
                    yield messages, self._format_state_memory()
                    return
                elif chunk[0] == "error":
                    continue
                elif chunk[0] == "stage-completed":
                    messages.append(gr.ChatMessage("assistant", f"[Iter {chunk[2]}] {chunk[1]} ..."))  # type: ignore
                    yield messages, self._format_state_memory()
                elif chunk[0] == "continue":
                    pass
                elif chunk[0] == "final":
                    messages.append(gr.ChatMessage("assistant", chunk[1]))
                    yield messages, self._format_state_memory()
                    return

    def _format_state_memory(self):
        return "\n".join([
            "## Feedbacks",
            self.bank.feedbacks_prompt,
            "## Codes",
            f"```python\n{self.bank.codes_prompt.lstrip() or 'Waiting for Response'}\n```",
            "## Instructions",
            self.bank.instructions_prompt,
            "## Variables",
            self.bank.variables_prompt,
        ])

    async def _call_vqa(
        self,
        image: bytes,
        query: str,
        state_memory_bank: StateMemoryBank,
        websocket: WebSocketClientProtocol
    ):
        with N.util.Timer(verbose=False) as timer:  # type: ignore
            # initial perception
            yield "start", "Reasoning"

            current_step_index = 0

            result, code = await self.reasoner.initial_run(image, query, websocket)
            t = timer.time(timer_msg := "Reasoner in Step 0")
            logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

            if result.type == "error":
                yield "panic", "Reasoning", result
                return

            state_memory_bank.extend_memory(
                result.feedbacks, [], [], result.variables, result.variable_names
            )
            guesses = []

            yield "stage-completed", "Reasoning", current_step_index, result, code

            for current_step_index in range(1, self.max_iterations + 1):
                # ----------------- Planner -----------------
                yield "start", "planning"
                instructions, probs = await self.planner(query, current_step_index, state_memory_bank)
                t = timer.time(timer_msg := f"Planner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if not instructions:
                    yield "panic", "Planning"
                    break

                # ----------------- Controller -----------------

                instruction = self.controller(instructions, probs)
                t = timer.time(timer_msg := f"Controller in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                yield "stage-completed", "Planning", current_step_index, sorted(zip(instructions, probs),
                    key=lambda x: x[1], reverse=True), instruction

                # ------------------ Reasoner ------------------
                yield "start", "Reasoning"

                result, code = await self.reasoner(image, query, instruction, current_step_index, state_memory_bank,
                    websocket)
                t = timer.time(timer_msg := f"Reasoner in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                if result.type == "error":
                    yield "error", "Reasoning", result
                    continue

                assert type(code) is str
                state_memory_bank.extend_memory(
                    result.feedbacks, [code], [instruction], result.variables, result.variable_names
                )
                # ------------------ Summarizer ------------------
                yield "stage-completed", "Reasoning", current_step_index, code

                yield "start", "Summarizing"

                guesses.append(await self.summarizer(query, state_memory_bank))
                t = timer.time(timer_msg := f"Summarizer in Step {current_step_index}")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                yield "stage-completed", "Summarizing", current_step_index

                if result.type == "continue":
                    yield "continue"
                    continue

                # ------------------ Summarizer ------------------ 
                final_result = await self.summarizer.final_guess(query, guesses)
                t = timer.time(timer_msg := f"Summarizer in Step Final")
                logger.debug(f"[Timer] {timer_msg}: {t:.4f} sec")

                yield "final", final_result
                return
