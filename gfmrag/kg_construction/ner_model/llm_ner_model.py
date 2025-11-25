"""
llm_ner_model.py - NER dùng Large Language Models

Sử dụng LLMs (GPT, Llama, etc.) để extract named entities thông qua prompting.
One-shot learning: Cho model 1 ví dụ để học format output.

Adapted from: https://github.com/OSU-NLP-Group/HippoRAG
"""

import logging
from typing import Literal

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.utils import extract_json_dict, processing_phrases

from .base_model import BaseNERModel

logger = logging.getLogger(__name__)
# Tắt logging của OpenAI và httpx để giảm noise
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# ============================================================================
# PROMPTS - One-shot learning
# ============================================================================

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""

query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""


class LLMNERModel(BaseNERModel):
    """
    NER model dùng LLMs với one-shot prompting.
    
    Cách hoạt động:
    1. System prompt: Define role (entity extraction system)
    2. One-shot example: Cho model 1 ví dụ input → output
    3. Actual query: Text cần extract entities
    4. Parse JSON response
    5. Clean entities với processing_phrases()
    
    Hỗ trợ LLM providers:
    - OpenAI: GPT-4, GPT-3.5 (với JSON mode)
    - Ollama: Local models
    - llama.cpp: Local GGUF models
    - Together/NVIDIA: API models
    
    Ví dụ:
        model = LLMNERModel("openai", "gpt-4o-mini")
        entities = model("Bill Gates founded Microsoft")
        # → ["bill gates", "microsoft"]
    """

    def __init__(
        self,
        llm_api: Literal["openai", "nvidia", "together", "ollama", "llama.cpp"] = "openai",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
    ):
        """
        Khởi tạo LLM-based NER model.
        
        Args:
            llm_api: LLM provider
                    - "openai": GPT models (có JSON mode)
                    - "ollama": Local Ollama models
                    - "llama.cpp": Local GGUF models
                    - "together"/"nvidia": API models
            model_name: Tên model cụ thể
                       - OpenAI: "gpt-4o-mini", "gpt-4"
                       - Ollama: "llama3", "mistral"
            max_tokens: Max tokens cho response (default 1024)
        """
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Khởi tạo LLM client qua langchain_util
        self.client = init_langchain_model(llm_api, model_name)

    def __call__(self, text: str) -> list:
        """
        Extract named entities từ text dùng LLM.
        
        Workflow:
        ┌─────────────────────────────────────────────────────┐
        │ 1. BUILD PROMPT (One-shot)                          │
        │    - System: "You're entity extraction system"      │
        │    - Example input/output                           │
        │    - Actual query                                   │
        ├─────────────────────────────────────────────────────┤
        │ 2. CALL LLM                                         │
        │    - OpenAI: JSON mode (structured output)         │
        │    - Others: Text mode → parse JSON                │
        ├─────────────────────────────────────────────────────┤
        │ 3. PARSE RESPONSE                                   │
        │    - Extract JSON dict                              │
        │    - Get "named_entities" field                     │
        ├─────────────────────────────────────────────────────┤
        │ 4. CLEAN ENTITIES                                   │
        │    - processing_phrases(): lowercase, remove chars  │
        │    - Return list of clean entities                  │
        └─────────────────────────────────────────────────────┘
        
        Args:
            text: Input text cần extract entities
                 Ví dụ: "Bill Gates founded Microsoft in Seattle"
        
        Returns:
            list: Cleaned entities
                 Ví dụ: ["bill gates", "microsoft", "seattle"]
                 Empty list [] nếu extraction fail
        
        Notes:
            - OpenAI có JSON mode → reliable structured output
            - Local models: Phải parse JSON manually từ text
            - Errors được catch và log, return []
        """
        # ====================================================================
        # BƯỚC 1: BUILD PROMPT (One-shot learning)
        # ====================================================================
        query_ner_prompts = ChatPromptTemplate.from_messages(
            [
                # System message: Định nghĩa role
                SystemMessage("You're a very effective entity extraction system."),
                
                # One-shot example (input)
                HumanMessage(query_prompt_one_shot_input),
                
                # One-shot example (output)
                AIMessage(query_prompt_one_shot_output),
                
                # Actual query
                HumanMessage(query_prompt_template.format(text)),
            ]
        )
        query_ner_messages = query_ner_prompts.format_prompt()

        # ====================================================================
        # BƯỚC 2: CALL LLM (khác nhau cho mỗi provider)
        # ====================================================================
        json_mode = False
        
        if isinstance(self.client, ChatOpenAI):
            # OpenAI: Dùng JSON mode cho structured output
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,                              # Deterministic
                max_tokens=self.max_tokens,
                stop=["\n\n"],                             # Stop tại double newline
                response_format={"type": "json_object"},   # Force JSON output
            )
            response_content = chat_completion.content
            json_mode = True
            
        elif isinstance(self.client, (ChatOllama, ChatLlamaCpp)):
            # Local models: Text mode, parse JSON manually
            response_content = self.client.invoke(query_ner_messages.to_messages())
            response_content = extract_json_dict(response_content)
            
        else:
            # Other API models: Text mode, parse JSON
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
            )
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)

        # ====================================================================
        # BƯỚC 3: VALIDATE RESPONSE
        # ====================================================================
        if not json_mode:
            # Nếu không phải JSON mode, validate response
            try:
                assert "named_entities" in response_content
                response_content = str(response_content)
            except Exception as e:
                print("Query NER exception", e)
                response_content = {"named_entities": []}

        # ====================================================================
        # BƯỚC 4: PARSE & CLEAN ENTITIES
        # ====================================================================
        try:
            # Parse JSON để lấy list entities
            ner_list = eval(response_content)["named_entities"]
            
            # Clean entities: lowercase, remove special chars
            query_ner_list = [processing_phrases(ner) for ner in ner_list]
            
            return query_ner_list
            
        except Exception as e:
            # Nếu có lỗi, log và return empty list
            logger.error(f"Error in extracting named entities: {e}")
            return []