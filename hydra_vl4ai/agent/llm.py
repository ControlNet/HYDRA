import asyncio
from functools import wraps
import os
import httpx
import numpy as np
from openai import AsyncOpenAI
from ollama import AsyncClient, Client
import openai
import time

from ..util.config import Config
from ..util.console import logger

try:
    openai_client = AsyncOpenAI()
except openai.OpenAIError:
    openai_client = None
    logger.debug("OpenAI API key is not set, ChatGPT will not work.")
else:
    logger.debug("OpenAI Client is set.")

try:
    ollama_client = AsyncClient(os.environ["OLLAMA_HOST"], timeout=120)
    # evaluate the connection
    Client(os.environ["OLLAMA_HOST"]).ps()
except httpx.ConnectError:
    ollama_client = None
    logger.debug("OLLAMA server is not available, Llama will not work.")
else:
    logger.debug("OLLAMA server is set.")

_semaphore = asyncio.Semaphore(Config.base_config["llm_max_concurrency"])


def handle_openai_exceptions(func):
    max_trial = Config.base_config["llm_max_retry"]

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


def handle_ollama_exceptions(func):
    max_trial = Config.base_config["llm_max_retry"]

    @wraps(func)
    async def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                return await func(*args, **kwargs)
            except httpx.ConnectError as e:
                pass
            except httpx.ConnectTimeout as e:
                pass
            except httpx.TimeoutException as e:
                pass
            except Exception as e:
                raise
    return wrapper


@handle_openai_exceptions
async def chatgpt(model_name: str, prompt: str):
    async with _semaphore:
        response = await openai_client.chat.completions.create(model=model_name,
            messages=[{"role": "user", "content": prompt}], timeout=30)
    return response.choices[0].message.content


@handle_openai_exceptions
async def gpt3_embedding(prompt: str):
    async with _semaphore:
        response = (await openai_client.embeddings.create(input = [prompt],
            model=Config.base_config["embedding_model"])).data[0].embedding
    response = np.array(response)
    return response


@handle_ollama_exceptions
async def ollama(model_name: str, prompt: str):
    async with _semaphore:
        response = await ollama_client.chat(model=model_name, 
            messages=[{"role": "user", "content": prompt}], stream=False, )
    return response["message"]["content"]


async def llm(model_name: str, prompt: str):
    if model_name.startswith("gpt"):
        return await chatgpt(model_name, prompt)
    elif model_name.startswith("llama") or model_name.startswith("deepseek-coder"):
        return await ollama(model_name, prompt)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
