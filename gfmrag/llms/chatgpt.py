import json
import logging
import os
import time

import dotenv
import requests
import tiktoken

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.ERROR)

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

OPENAI_MODEL = ["gpt-4", "gpt-3.5-turbo"]


def get_token_limit(model: str = "gpt-4") -> int:
    """Returns the token limitation of provided model"""
    if model in ["gpt-4", "gpt-4-0613"]:
        num_tokens_limit = 8192
    elif model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
        num_tokens_limit = 128000
    elif model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]:
        num_tokens_limit = 16384
    elif model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "text-davinci-003",
        "text-davinci-002",
    ]:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(
            f"""get_token_limit() is not implemented for model {model}."""
        )
    return num_tokens_limit


class ChatGPT(BaseLanguageModel):
    """A class that interacts with ChatGPT-compatible APIs (YEScale, OpenAI, etc.) using requests library.

    This class provides functionality to generate text using ChatGPT models while handling
    token limits, retries, and various input formats. Supports YEScale and OpenAI APIs.

    Args:
        model_name_or_path (str): The name or path of the ChatGPT model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5
        api_key (str, optional): API key for authentication. If not provided, will use
            YESCALE_API_KEY or OPENAI_API_KEY from environment variables
        api_url (str, optional): Full API URL (including /chat/completions). If not provided,
            will use YESCALE_API_BASE_URL from environment or default to OpenAI's URL

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the ChatGPT model being used
        maximun_token (int): Maximum token limit for the specified model
        api_key (str): API key for authentication
        api_url (str): Full API endpoint URL

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the ChatGPT model

    Raises:
        KeyError: If the specified model is not found when calculating tokens
        Exception: If generation fails after maximum retries
    """

    def __init__(
        self,
        model_name_or_path: str,
        retry: int = 5,
        api_key: str | None = None,
        api_url: str | None = None,
    ):
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token = get_token_limit(self.model_name)

        # Priority: 1. Explicit params, 2. YEScale env vars, 3. OpenAI env vars
        if api_key is None:
            api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )

        if api_url is None:
            # Try YEScale first, then OpenAI default
            api_url = os.environ.get("YESCALE_API_BASE_URL") or "https://api.openai.com/v1/chat/completions"

        self.api_key = api_key
        self.api_url = api_url
        logger.info(f"Using API endpoint: {self.api_url}")

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except KeyError as e:
            raise KeyError(f"Warning: model {self.model_name} not found.") from e
        return num_tokens

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the ChatGPT API.

        This method sends a request to the ChatGPT API and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input length {input_length} is too long. The maximum token is {self.maximun_token}.\n Right truncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                # Use requests library to call API (YEScale compatible)
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }

                payload = {
                    "model": self.model_name,
                    "messages": message,
                    "temperature": 0.0
                }
                # Don't send max_tokens to let API decide based on model capabilities
                # YEScale may have different limits than OpenAI

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60
                )

                # Check response status
                if response.status_code == 200:
                    data = response.json()
                    result = data['choices'][0]['message']['content'].strip()
                    return result
                else:
                    raise Exception(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                logger.error(f"Message: {llm_input}")
                logger.error(f"Number of tokens: {self.token_len(message_string)}")
                logger.error(f"Error: {e}")
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
