"""
YEScale Chat Model - Custom LangChain model for YEScale API

Lý do tạo custom model:
- OpenAI SDK luôn tự động append '/chat/completions' vào base_url
- YEScale API endpoint là '/v1/chat/completion' (không có 's')
- Không thể config OpenAI SDK để không append
- Giải pháp: Dùng requests library trực tiếp như ChatGPT class

Usage:
    from gfmrag.kg_construction.yescale_chat_model import YEScaleChatModel

    model = YEScaleChatModel(
        api_url="https://api.yescale.io/v1/chat/completion",
        api_key="sk-xxx",
        model="gpt-4o-mini"
    )
    response = model.invoke("Hello!")
"""

import json
import os
from typing import Any, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class YEScaleChatModel(BaseChatModel):
    """
    Custom LangChain chat model for YEScale API using requests library.

    Attributes:
        api_url: Full API endpoint URL (e.g., "https://api.yescale.io/v1/chat/completion")
        api_key: YEScale API key
        model: Model name (e.g., "gpt-4o-mini")
        temperature: Temperature for sampling (default: 0.0)
        max_retries: Number of retries on failure (default: 5)
        timeout: Request timeout in seconds (default: 60)
    """

    api_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_retries: int = 5
    timeout: int = 60

    class Config:
        """Pydantic config for LangChain compatibility."""
        arbitrary_types_allowed = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completion using YEScale API.

        Args:
            messages: List of LangChain messages
            stop: Stop sequences (not used)
            run_manager: Callback manager (not used)
            **kwargs: Additional parameters

        Returns:
            ChatResult with generated message

        Raises:
            requests.HTTPError: If API request fails
        """
        # Convert LangChain messages to OpenAI format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback for other message types
                formatted_messages.append({"role": "user", "content": str(msg.content)})

        # Prepare request
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }

        # Add stop sequences if provided
        if stop:
            payload["stop"] = stop

        # Make request with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse response
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()

                # Return ChatResult
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Retry on failure
                    continue
                else:
                    # Last attempt failed, raise exception
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Failed to generate response after retries")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "yescale"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for this LLM."""
        return {
            "model": self.model,
            "api_url": self.api_url,
            "temperature": self.temperature,
        }


def create_yescale_model(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> YEScaleChatModel:
    """
    Factory function to create YEScaleChatModel with environment variable fallback.

    Args:
        api_url: Full API endpoint URL (defaults to YESCALE_API_BASE_URL env var)
        api_key: API key (defaults to YESCALE_API_KEY or OPENAI_API_KEY env var)
        model: Model name
        temperature: Sampling temperature
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds
        **kwargs: Additional parameters

    Returns:
        YEScaleChatModel instance

    Example:
        model = create_yescale_model(
            api_url="https://api.yescale.io/v1/chat/completion",
            api_key="sk-xxx",
            model="gpt-4o-mini"
        )
    """
    # Get from env vars if not provided
    if api_url is None:
        api_url = os.environ.get("YESCALE_API_BASE_URL")
        if api_url is None:
            raise ValueError("api_url must be provided or YESCALE_API_BASE_URL must be set")

    if api_key is None:
        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("api_key must be provided or YESCALE_API_KEY/OPENAI_API_KEY must be set")

    return YEScaleChatModel(
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )
