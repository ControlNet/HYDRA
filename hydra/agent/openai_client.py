import asyncio
from functools import wraps
import numpy as np
from openai import AsyncOpenAI
import openai
from dotenv import load_dotenv
import time

from ..util.config import Config

load_dotenv()
openai_client = AsyncOpenAI()

_semaphore = asyncio.Semaphore(Config.base_config["openai_max_concurrency"])


def handle_openai_exceptions(func):
    max_trial = Config.base_config["openai_max_retry"]

    @wraps(func)
    async def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                return await func(*args, **kwargs)
            except openai.APITimeoutError as e:
                pass
            except openai.APIConnectionError as e:
                pass
            except openai.RateLimitError as e:
                time.sleep(1)
                pass
            except openai.BadRequestError as e:
                # maybe exceed the length, should raise directly
                raise
            except openai.APIStatusError as e:
                # server side problem, should raise directly
                raise
            except Exception as e:
                raise
    return wrapper


@handle_openai_exceptions
async def chatgpt(prompt: str):
    async with _semaphore:
        response = await openai_client.chat.completions.create(model=Config.base_config["llm_model"],
            messages=[{"role": "user", "content": prompt}], timeout=30)
    return response.choices[0].message.content


@handle_openai_exceptions
async def gpt3_embedding(prompt: str):
    async with _semaphore:
        response = (await openai_client.embeddings.create(input = [prompt],
            model=Config.base_config["embedding_model"])).data[0].embedding
    response = np.array(response)
    return response
