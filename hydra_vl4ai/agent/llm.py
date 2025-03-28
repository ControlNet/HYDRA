import asyncio
import os
import time
from functools import wraps
import os
from typing import Literal
import httpx
import numpy as np
from openai import NOT_GIVEN, AsyncOpenAI, AsyncAzureOpenAI
from ollama import AsyncClient, Client
import openai
import time
from tensorneko_util.util import Singleton

from ..util.config import Config
from ..util.console import logger


@Singleton
class Cost:
    def __init__(self):
        self.cost = 0
        self.input_tokens = 0
        self.output_tokens = 0
    
    def add(self, chatgpt_response, model_name):
        if model_name.startswith("gpt-4o-mini"):
            input_pricing = 0.15 / 1000000
            output_pricing = 0.6 / 1000000
        elif model_name.startswith("gpt-4o"):
            input_pricing = 2.5 / 1000000
            output_pricing = 10 / 1000000
        elif model_name.startswith("gpt-35-turbo"):
            input_pricing = 0.5 / 1000000
            output_pricing = 1.5 / 1000000
        else:
            input_pricing = 0
            output_pricing = 0

        price = chatgpt_response.usage.prompt_tokens * input_pricing + chatgpt_response.usage.completion_tokens * output_pricing
        self.cost += price
        self.input_tokens += chatgpt_response.usage.prompt_tokens
        self.output_tokens += chatgpt_response.usage.completion_tokens

try:
    if os.environ.get("AZURE_OPENAI_URL", "") != "":
        logger.debug("Azure OpenAI API key is set.")
        openai_client = AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_URL"],
            api_version="2023-07-01-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=30.
        )
    else:
        openai_client = AsyncOpenAI(timeout=30)
except openai.OpenAIError:
    openai_client = None
    logger.debug("OpenAI API key is not set, ChatGPT will not work.")
else:
    logger.debug("OpenAI Client is set.")

try:
    ollama_client = AsyncClient(os.environ["OLLAMA_HOST"], timeout=120)
    # evaluate the connection
    Client(os.environ["OLLAMA_HOST"]).ps()
except (httpx.ConnectError, ConnectionError):
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
                logger.debug(f"Call OpenAI API. {args, kwargs}")
                return await func(*args, **kwargs)
            except openai.APITimeoutError as e:
                logger.error(f"OpenAI API Timeout: {e}")
                pass
            except openai.APIConnectionError as e:
                logger.error(f"OpenAI API Connection Error: {e}")
                pass
            except openai.RateLimitError as e:
                logger.error(f"OpenAI Rate Limit Error: {e}")
                time.sleep(1)
                pass
            except openai.BadRequestError as e:
                # maybe exceed the length, should raise directly
                raise e
            except openai.APIStatusError as e:
                # server side problem, should raise directly
                raise e
            except Exception as e:
                raise
            logger.debug(f"Retry OpenAI API call.")
    return wrapper


def handle_ollama_exceptions(func):
    max_trial = Config.base_config["llm_max_retry"]

    @wraps(func)
    async def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                return await func(*args, **kwargs)
            except httpx.ConnectError:
                pass
            except httpx.ConnectTimeout:
                pass
            except httpx.TimeoutException:
                pass
            except Exception as e:
                raise e

    return wrapper


@handle_openai_exceptions
async def chatgpt(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str:
    response_format = {"type": "json_object"} if format == "json" else NOT_GIVEN
    async with _semaphore:
        response = await openai_client.chat.completions.create(model=model_name,
            messages=messages, response_format=response_format)
        Cost.add(response, model_name)
    return response.choices[0].message.content


@handle_openai_exceptions
async def gpt3_embedding(model_name: str, prompt: str):
    async with _semaphore:
        response = (await openai_client.embeddings.create(input=[prompt],
            model=model_name)).data[0].embedding
    response = np.array(response)
    return response


@handle_ollama_exceptions
async def ollama(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str:
    async with _semaphore:
        response = await ollama_client.chat(model=model_name, 
            messages=messages, stream=False, format=format)
    return response["message"]["content"]


async def llm(model_name: str, prompt: str, format: Literal["", "json"] = "") -> str:
    if model_name.startswith("gpt"):
        return await chatgpt(model_name, [{"role": "user", "content": prompt}], format)
    else:
        return await ollama(model_name, [{"role": "user", "content": prompt}], format)


async def llm_embedding(model_name: str, prompt: str):
    if model_name in ("text-embedding-3-small", "text-embedding-3-large"):
        return await gpt3_embedding(model_name, prompt)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


async def llm_with_message(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str:
    if model_name.startswith("gpt"):
        return await chatgpt(model_name, messages, format)
    else:
        return await ollama(model_name, messages, format)
