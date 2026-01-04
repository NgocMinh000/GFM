"""
llm_openie_model.py - OpenIE dùng Large Language Models

Pipeline 2 bước:
1. NER: Extract named entities từ text
2. Relation Extraction: Extract triples (subject, relation, object) từ entities

Adapted from: https://github.com/OSU-NLP-Group/HippoRAG
"""

import json
import logging
from itertools import chain
from typing import Literal

import numpy as np
from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from gfmrag.kg_construction.utils import extract_json_dict

from .base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)
# Tắt logging của OpenAI và httpx
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class LLMOPENIEModel(BaseOPENIEModel):
    """
    OpenIE model dùng LLMs với 2-step pipeline.
    
    Pipeline:
    ┌─────────────────────────────────────────────────────┐
    │ Step 1: NER (Named Entity Recognition)             │
    │ Input: "Bill Gates founded Microsoft in 1975"      │
    │ Output: ["Bill Gates", "Microsoft", "1975"]        │
    ├─────────────────────────────────────────────────────┤
    │ Step 2: Relation Extraction                        │
    │ Input: Text + Entities                             │
    │ Output: [                                          │
    │   ["Bill Gates", "founded", "Microsoft"],         │
    │   ["Microsoft", "founded in", "1975"]             │
    │ ]                                                   │
    └─────────────────────────────────────────────────────┘
    
    Hỗ trợ:
    - OpenAI: JSON mode cho structured output
    - Ollama/llama.cpp: Local models
    - Together/NVIDIA: API models
    
    Ví dụ:
        model = LLMOPENIEModel("openai", "gpt-4o-mini")
        result = model("Bill Gates founded Microsoft")
        # → {
        #     "passage": "Bill Gates founded Microsoft",
        #     "extracted_entities": ["Bill Gates", "Microsoft"],
        #     "extracted_triples": [["Bill Gates", "founded", "Microsoft"]]
        # }
    """

    def __init__(
        self,
        llm_api: Literal["openai", "nvidia", "together", "ollama", "llama.cpp"] = "openai",
        model_name: str = "gpt-4o-mini",
        max_ner_tokens: int = 1024,
        max_triples_tokens: int = 4096,
    ):
        """
        Khởi tạo LLM-based OpenIE model.
        
        Args:
            llm_api: LLM provider
            model_name: Tên model
            max_ner_tokens: Max tokens cho NER output (default 1024)
            max_triples_tokens: Max tokens cho triples output (default 4096)
                               Cao hơn vì triples thường dài hơn entities
        """
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_ner_tokens = max_ner_tokens
        self.max_triples_tokens = max_triples_tokens

        # Khởi tạo LLM client
        self.client = init_langchain_model(llm_api, model_name)

    def ner(self, text: str) -> list:
        """
        Step 1: Named Entity Recognition.
        
        Extract named entities từ text sử dụng LLM với few-shot prompts.
        
        Args:
            text: Input text
        
        Returns:
            list: Named entities
                 Ví dụ: ["Bill Gates", "Microsoft", "1975"]
                 Empty list [] nếu fail
        
        Workflow:
        1. Format prompt với few-shot examples (từ ner_prompts)
        2. Call LLM
        3. Parse JSON response để lấy "named_entities" field
        4. Return list entities
        
        Notes:
        - OpenAI: Dùng JSON mode → reliable
        - Others: Parse JSON manually từ text
        """
        # Format NER prompt với few-shot examples
        ner_messages = ner_prompts.format_prompt(user_input=text)

        try:
            if isinstance(self.client, ChatOpenAI):
                # OpenAI: JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_ner_tokens,
                    stop=["\n\n"],
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content
                response_content = eval(response_content)

            elif isinstance(self.client, (ChatOllama, ChatLlamaCpp)):
                # Local models: Parse JSON từ text
                response_content = self.client.invoke(
                    ner_messages.to_messages()
                ).content
                response_content = extract_json_dict(response_content)

            else:
                # Other models: Parse JSON
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(), 
                    temperature=0
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)

            # Extract entities từ JSON response
            if "named_entities" not in response_content:
                logger.warning(
                    f"NER response missing 'named_entities' field. "
                    f"Response: {response_content}. "
                    f"Text preview: {text[:100]}..."
                )
                response_content = []
            else:
                response_content = response_content["named_entities"]

        except Exception as e:
            logger.error(
                f"Error in extracting named entities: {e}. "
                f"Text preview: {text[:100]}..."
            )
            response_content = []

        return response_content

    def openie_post_ner_extract(self, text: str, entities: list) -> str:
        """
        Step 2: Relation Extraction post-NER.
        
        Extract triples (subject, relation, object) từ text với entities đã biết.
        
        Args:
            text: Input text
            entities: Entities đã extract từ NER
                     Ví dụ: ["Bill Gates", "Microsoft"]
        
        Returns:
            str: JSON string chứa triples
                Format: '{"triples": [["subj", "rel", "obj"], ...]}'
                Return "{}" nếu fail
        
        Workflow:
        1. Format prompt với text + entities
        2. Call LLM để extract relations
        3. Return JSON string (chưa parse)
        
        Notes:
        - Input entities giúp model focus vào đúng entities
        - Triples có thể overlap với entities (1 entity làm cả subject và object)
        """
        # Format entities thành JSON
        named_entity_json = {"named_entities": entities}
        
        # Format OpenIE prompt với text + entities
        openie_messages = openie_post_ner_prompts.format_prompt(
            passage=text, 
            named_entity_json=json.dumps(named_entity_json)
        )
        
        try:
            if isinstance(self.client, ChatOpenAI):
                # OpenAI: JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content

            elif isinstance(self.client, (ChatOllama, ChatLlamaCpp)):
                # Local models: Parse JSON
                response_content = self.client.invoke(
                    openie_messages.to_messages()
                ).content
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)
                
            else:
                # Other models: Parse JSON
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)

        except Exception as e:
            logger.error(f"Error in OpenIE: {e}")
            response_content = "{}"

        return response_content

    def __call__(self, text: str) -> dict:
        """
        Thực hiện complete OpenIE pipeline.
        
        Full workflow:
        ┌──────────────────────────────────────────────────┐
        │ Input: "Bill Gates founded Microsoft in 1975"   │
        ├──────────────────────────────────────────────────┤
        │ Step 1: NER                                      │
        │ → entities: ["Bill Gates", "Microsoft", "1975"] │
        ├──────────────────────────────────────────────────┤
        │ Step 2: Flatten nested lists (nếu có)           │
        │ → entities: ["Bill Gates", "Microsoft", "1975"] │
        ├──────────────────────────────────────────────────┤
        │ Step 3: Relation Extraction                     │
        │ Input: text + entities                          │
        │ → triples: [["Bill Gates", "founded",           │
        │              "Microsoft"],                       │
        │             ["Microsoft", "founded in", "1975"]] │
        ├──────────────────────────────────────────────────┤
        │ Output: {                                        │
        │   "passage": "...",                             │
        │   "extracted_entities": [...],                  │
        │   "extracted_triples": [...]                    │
        │ }                                                │
        └──────────────────────────────────────────────────┘
        
        Args:
            text: Input text cần extract
        
        Returns:
            dict: {
                "passage": str,              # Input text
                "extracted_entities": list,  # Entities
                "extracted_triples": list    # Triples
            }
        
        Notes:
        - Nếu NER fail → warning log, triples vẫn chạy với empty entities
        - Nếu relation extraction fail → extracted_triples = []
        - Flatten nested lists để handle edge cases
        """
        # Initialize result
        res = {
            "passage": text, 
            "extracted_entities": [], 
            "extracted_triples": []
        }

        # ================================================================
        # STEP 1: NER
        # ================================================================
        doc_entities = self.ner(text)
        
        # Flatten nested lists nếu có (edge case)
        try:
            doc_entities = list(np.unique(doc_entities))
        except Exception as e:
            logger.error(f"Results has nested lists: {e}")
            # chain.from_iterable: Flatten nested lists
            doc_entities = list(np.unique(list(chain.from_iterable(doc_entities))))
        
        # Warning nếu không extract được entities
        if not doc_entities:
            logger.warning(
                f"No entities extracted. Possibly model not following instructions. "
                f"Text preview: {text[:200]}..."
            )
        
        # ================================================================
        # STEP 2: RELATION EXTRACTION
        # ================================================================
        triples = self.openie_post_ner_extract(text, doc_entities)
        
        # ================================================================
        # STEP 3: FORMAT RESULT
        # ================================================================
        res["extracted_entities"] = doc_entities
        
        # Parse triples JSON
        try:
            res["extracted_triples"] = eval(triples)["triples"]
        except Exception:
            logger.error(f"Error in parsing triples: {triples}")
            # Nếu parse fail, extracted_triples = [] (đã init)

        return res