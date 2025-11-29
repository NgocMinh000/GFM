"""
langchain_util.py - Khởi tạo LangChain Language Models

Chức năng: Factory function để tạo LLM instances từ các providers khác nhau
Hỗ trợ: OpenAI, NVIDIA, Together, Ollama, llama.cpp
"""

import os
from typing import Any

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from gfmrag.kg_construction.yescale_chat_model import YEScaleChatModel


def init_langchain_model(
    llm: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
    **kwargs: Any,
) -> ChatOpenAI | ChatTogether | ChatOllama | ChatLlamaCpp | YEScaleChatModel:
    """
    Khởi tạo language model từ LangChain.
    
    Args:
        llm: Provider ("openai", "nvidia", "together", "ollama", "llama.cpp")
        model_name: Tên model (ví dụ: "gpt-4", "llama3")
        temperature: Độ sáng tạo (0.0-2.0)
        max_retries: Số lần retry khi API fail
        timeout: Timeout cho mỗi API call (giây)
        **kwargs: Tham số bổ sung (max_tokens, top_p, etc.)
    
    Returns:
        LangChain chat model instance
    
    Ví dụ:
        llm = init_langchain_model("openai", "gpt-4", temperature=0.7)
        response = llm.invoke("Hello!")
    """
    
    if llm == "openai":
        # OpenAI GPT models or YEScale compatible API
        # Requires: OPENAI_API_KEY or YESCALE_API_KEY environment variable
        # Optional: YESCALE_API_BASE_URL for custom API endpoint
        assert model_name.startswith("gpt-"), f"OpenAI model must start with 'gpt-', got {model_name}"

        # Priority: 1. YEScale env vars, 2. OpenAI env vars
        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        yescale_url = os.environ.get("YESCALE_API_BASE_URL")

        # If YEScale URL is configured, use custom YEScaleChatModel (no URL manipulation)
        if yescale_url:
            print(f"[LangChain] Using YEScale API at: {yescale_url} (full URL, no appending)")
            return YEScaleChatModel(
                api_url=yescale_url,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                **kwargs,
            )
        else:
            # Use standard OpenAI SDK for official OpenAI API
            print(f"[LangChain] Using OpenAI API (official)")
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                **kwargs,
            )
    
    elif llm == "nvidia":
        # NVIDIA AI Endpoints
        # Requires: NVIDIA_API_KEY environment variable
        return ChatNVIDIA(
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    
    elif llm == "together":
        # Together AI
        # Requires: TOGETHER_API_KEY environment variable
        return ChatTogether(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            model=model_name,
            temperature=temperature,
            **kwargs,
        )
    
    elif llm == "ollama":
        # Local Ollama (không cần API key)
        # Requires: Ollama server running (ollama serve)
        return ChatOllama(model=model_name)
    
    elif llm == "llama.cpp":
        # Local llama.cpp (không cần API key)
        # model_name = path to GGUF file
        return ChatLlamaCpp(model_path=model_name, verbose=True)
    
    else:
        # Provider chưa được implement
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")