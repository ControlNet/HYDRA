from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import argparse
import traceback
from functools import partial
from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--external_packages", type=list, nargs="+", default=[])
args = parser.parse_args()

from .util.config import Config
Config.base_config_path = args.base_config
Config.model_config_path = args.model_config

from .execution.image_patch import *
from .execution.image_patch import ImagePatch
from .agent.smb.state_memory_bank import StateMemoryBank
from .execution.toolbox import Toolbox
from .util.message import ExecutionRequest, ExecutionResult
from .util.misc import get_description_from_executed_variable_list, get_statement_variable, load_image_from_bytes


def run_init():
    try:
        Toolbox.init(args.external_packages)
    except Exception as e:
        print(f"Error initializing toolbox: {e}")
        exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(asyncio.to_thread(run_init))
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/is_loaded")
async def is_loaded():
    return {"result": Toolbox.inited}


@app.websocket("/ws/")
async def execute(websocket: WebSocket):
    await websocket.accept()
    temp_state_memory_bank = StateMemoryBank()
    while True:
        temp_state_memory_bank.reset()
        try:
            execution_request = ExecutionRequest.from_dict(await websocket.receive_json())
        except WebSocketDisconnect:
            break
        if execution_request.send_image:
            image_buffer = await websocket.receive_bytes()
            image_patch = ImagePatch(load_image_from_bytes(image_buffer), state_memory_bank=temp_state_memory_bank)
        try:
            # run in new thread
            await asyncio.to_thread(exec, execution_request.code, globals(), locals())
        except Exception as e:
            # print traceback
            traceback.print_exc()
            result = ExecutionResult(
                "error",
                [],
                [],
                [],
                str(e)
            )
        else:
            new_variable_names = get_statement_variable(execution_request.code, locals())
            new_variable_and_description = get_description_from_executed_variable_list(new_variable_names, locals())

            # check if it is the final
            is_final = False
            if 'final_answer' in locals(): 
                output = locals()['final_answer']

                match Config.base_config["task"]:
                    case "grounding":
                        if isinstance(output, ImagePatch):
                            is_final = True
                            output = {"final_answer": [output]}
                            
                        elif isinstance(output, dict):
                            is_final = True
                            if all(isinstance(v, ImagePatch) for v in output.values()):
                                output = {k: [v] for k, v in output.items()}
                            elif all(isinstance(v, list) and all(isinstance(i, ImagePatch) for i in v) for v in output.values()):
                                pass
                            else:
                                is_final = False

                        elif isinstance(output, list) and len(output) == 1:
                            is_final = True
                            if all(isinstance(v, ImagePatch) for v in output):
                                output = {"final_answer": output}
                            else:
                                is_final = False

                        else:
                            temp_state_memory_bank.return_final_answer_should_be_ImagePatch()
                    case "vqa":
                        # if it is vqa, the final answer should be string
                        if isinstance(output, str):
                            is_final = True
                        else:
                            is_final = False
                            temp_state_memory_bank.return_final_answer_should_be_string()

            if is_final:
                
                match Config.base_config["task"]:
                    case "grounding":
                        return_result = {k: [img.to_bbox() for img in v] for k, v in output.items()}
                    case "vqa":
                        return_result = {"final_answer": output}

                result = ExecutionResult(
                    "final", 
                    temp_state_memory_bank.feedbacks,
                    new_variable_and_description,
                    new_variable_names,
                    json.dumps(return_result)
                )
            else:
                result = ExecutionResult(
                    "continue", 
                    temp_state_memory_bank.feedbacks,
                    new_variable_and_description,
                    new_variable_names,
                    ""
                )

        await websocket.send_json(result.to_dict())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.base_config["executor_port"], reload=False, workers=1)
