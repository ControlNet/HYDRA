import asyncio
import os
import time
from functools import wraps
import re
from typing import Literal
import httpx
import numpy as np
from openai import NOT_GIVEN, AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI, OpenAI
from ollama import AsyncClient, Client
import openai
import time
from tensorneko_util.util import Singleton

from ..util.config import Config
from ..util.console import logger


def parse_model_name(model_spec: str) -> tuple[str, str]:
    # Used to parse the model specification string
    # Example: "ollama::deepseek-r1:70b" -> ("ollama", "deepseek-r1:70b")
    assert "::" in model_spec, "Model specification must contain '::' to separate api_type and model_name"
    api_type, model_name = model_spec.split("::", 1)
    return api_type.lower(), model_name


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
        openai_client_sync = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_URL"],
            api_version="2023-07-01-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=30.
        )
    else:
        openai_client = AsyncOpenAI(timeout=30)
        openai_client_sync = OpenAI(timeout=30)
except openai.OpenAIError:
    openai_client = None
    openai_client_sync = None
    logger.debug("OpenAI API key is not set, ChatGPT will not work.")
else:
    logger.debug("OpenAI Client is set.")

try:
    ollama_client = AsyncClient(os.environ["OLLAMA_HOST"], timeout=120)
    ollama_client_sync = Client(os.environ["OLLAMA_HOST"])
    # evaluate the connection
    ollama_client_sync.ps()
except (httpx.ConnectError, ConnectionError):
    ollama_client = None
    ollama_client_sync = None
    logger.debug("OLLAMA server is not available, Llama will not work.")
else:
    logger.debug(f"OLLAMA server is set on {os.environ['OLLAMA_HOST']}")

# Set up vLLM client (using OpenAI client with custom base URL)
try:
    vllm_host = os.environ.get("VLLM_HOST", "")
    if vllm_host != "":
        # Get API key from environment variable if set
        vllm_api_key = os.environ.get("VLLM_API_KEY", "")
        
        vllm_client = AsyncOpenAI(
            base_url=vllm_host,
            api_key=vllm_api_key,
            timeout=60.0
        )
        vllm_client_sync = OpenAI(
            base_url=vllm_host,
            api_key=vllm_api_key,
            timeout=60.0
        )
        vllm_available = True
        logger.debug(f"vLLM client is initialized with server at {vllm_host}")
        if vllm_api_key != "":
            logger.debug("vLLM API key is set")
    else:
        vllm_client = None
        vllm_client_sync = None
        vllm_available = False
        logger.debug("VLLM_HOST environment variable is not set, vLLM will not work.")
except Exception as e:
    vllm_client = None
    vllm_client_sync = None
    vllm_available = False
    logger.debug(f"Error setting up vLLM client: {e}")

_semaphore = asyncio.Semaphore(Config.base_config["llm_max_concurrency"])


def handle_openai_exceptions(func):
    max_trial = Config.base_config["llm_max_retry"]

    @wraps(func)
    async def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                logger.debug(f"Call OpenAI API. {args, kwargs}")
                response = await func(*args, **kwargs)
                logger.debug(f"Response: {response}")
                return response
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
                response = await func(*args, **kwargs)
                logger.debug(f"Response: {response}")
                return response
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
async def chatgpt(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    response_format = {"type": "json_object"} if format == "json" else NOT_GIVEN
    async with _semaphore:
        response = await openai_client.chat.completions.create(model=model_name,
            messages=messages, response_format=response_format)
        Cost.add(response, model_name)
        logger.debug(f"Cost: {Cost.cost:.4f}, input: {Cost.input_tokens}, output: {Cost.output_tokens}")
    return response.choices[0].message.content


@handle_openai_exceptions
async def gpt3_embedding(model_name: str, prompt: str):
    async with _semaphore:
        response = (await openai_client.embeddings.create(input=[prompt], model=model_name)).data[0].embedding
    response = np.array(response)
    return response


@handle_ollama_exceptions
async def ollama(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str:
    # set non-thinking mode
    if "qwen3" in model_name.lower():
        messages[-1]["content"] += "/no_think"

    async with _semaphore:
        response = await ollama_client.chat(model=model_name, messages=messages, stream=False, format=format)
    response_content = response["message"]["content"]

    if "qwen3" in model_name.lower():
        # remove the <think> part
        pattern = r"<think>\s*?</think>"
        response_content = re.sub(pattern, "", response_content, flags=re.DOTALL)
        response_content = response_content.lstrip("\n")
    return response_content


@handle_openai_exceptions
async def vllm(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    if not vllm_available:
        raise ValueError("vLLM is not available. Set VLLM_HOST environment variable.")
    
    response_format = {"type": "json_object"} if format == "json" else NOT_GIVEN
    
    async with _semaphore:
        response = await vllm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format=response_format
        )
    
    return response.choices[0].message.content


async def llm(model_spec: str, prompt: str, format: Literal["", "json"] = "") -> str | None:
    api_type, model_name = parse_model_name(model_spec)
    
    if api_type == "openai":
        return await chatgpt(model_name, [{"role": "user", "content": prompt}], format)
    elif api_type == "ollama":
        return await ollama(model_name, [{"role": "user", "content": prompt}], format)
    elif api_type == "vllm":
        return await vllm(model_name, [{"role": "user", "content": prompt}], format)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


async def llm_embedding(model_spec: str, prompt: str):
    api_type, model_name = parse_model_name(model_spec)
    
    if api_type == "openai" and model_name in ("text-embedding-3-small", "text-embedding-3-large"):
        return await gpt3_embedding(model_name, prompt)
    else:
        raise ValueError(f"Model {model_spec} is not supported for embeddings.")


async def llm_with_message(model_spec: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    api_type, model_name = parse_model_name(model_spec)
    
    if api_type == "openai":
        return await chatgpt(model_name, messages, format)
    elif api_type == "ollama":
        return await ollama(model_name, messages, format)
    elif api_type == "vllm":
        return await vllm(model_name, messages, format)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


# sync versions
def handle_openai_exceptions_sync(func):
    max_trial = Config.base_config["llm_max_retry"]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                logger.debug(f"Call OpenAI API. {args, kwargs}")
                return func(*args, **kwargs)
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


def handle_ollama_exceptions_sync(func):
    max_trial = Config.base_config["llm_max_retry"]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for _ in range(max_trial):
            try:
                return func(*args, **kwargs)
            except httpx.ConnectError:
                pass
            except httpx.ConnectTimeout:
                pass
            except httpx.TimeoutException:
                pass
            except Exception as e:
                raise e

    return wrapper
    

def llm_sync(model_spec: str, prompt: str, format: Literal["", "json"] = "") -> str | None:
    api_type, model_name = parse_model_name(model_spec)
    
    if api_type == "openai":
        return chatgpt_sync(model_name, [{"role": "user", "content": prompt}], format)
    elif api_type == "ollama":
        return ollama_sync(model_name, [{"role": "user", "content": prompt}], format)
    elif api_type == "vllm":
        return vllm_sync(model_name, [{"role": "user", "content": prompt}], format)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


@handle_openai_exceptions_sync
def chatgpt_sync(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    response_format = {"type": "json_object"} if format == "json" else NOT_GIVEN
    response = openai_client_sync.chat.completions.create(model=model_name,
            messages=messages, response_format=response_format)
    Cost.add(response, model_name)
    logger.debug(f"Cost: {Cost.cost:.4f}, input: {Cost.input_tokens}, output: {Cost.output_tokens}")
    return response.choices[0].message.content


@handle_openai_exceptions_sync
def ollama_sync(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    response = ollama_client_sync.chat(model=model_name, messages=messages, stream=False, format=format)
    response_content = response["message"]["content"]

    if "qwen3" in model_name.lower():
        # remove the <think> part
        pattern = r"<think>\s*?</think>"
        response_content = re.sub(pattern, "", response_content, flags=re.DOTALL)
        response_content = response_content.lstrip("\n")
    return response_content


@handle_openai_exceptions_sync
def vllm_sync(model_name: str, messages: list[dict[str, str]], format: Literal["", "json"] = "") -> str | None:
    if not vllm_available:
        raise ValueError("vLLM is not available. Set VLLM_HOST environment variable.")
    
    response_format = {"type": "json_object"} if format == "json" else NOT_GIVEN
    
    response = vllm_client_sync.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format=response_format
    )
    
    return response.choices[0].message.content
