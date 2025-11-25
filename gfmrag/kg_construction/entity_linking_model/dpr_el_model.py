"""
dpr_el_model.py - Entity Linking dùng Dense Passage Retrieval

DPR: Encode entities thành dense vectors, tính similarity bằng dot product
- Single-vector representation cho mỗi entity
- Fast similarity search với matrix multiplication
- Support caching embeddings
"""

import hashlib
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .base_model import BaseELModel


class DPRELModel(BaseELModel):
    """
    Entity Linking model dùng Dense Passage Retrieval (DPR).
    
    DPR approach:
    - Encode entities thành dense vectors (single embedding per entity)
    - Similarity = cosine similarity (dot product nếu normalized)
    - Cache embeddings để tránh recompute
    
    Workflow:
    1. index(): Encode entities → cache embeddings
    2. __call__(): Encode queries → compute similarity → return top-k
    
    Ví dụ:
        model = DPRELModel('sentence-transformers/all-mpnet-base-v2')
        model.index(['Paris', 'London', 'Berlin'])
        
        results = model(['paris city'], topk=2)
        # → {'paris city': [
        #       {'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
        #       {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}
        #    ]}
    """

    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        """
        Khởi tạo DPR Entity Linking model.
        
        Args:
            model_name: SentenceTransformer model name hoặc path
                       Ví dụ: "sentence-transformers/all-mpnet-base-v2"
            root: Thư mục cache embeddings
            use_cache: Cache embeddings để tránh recompute
            normalize: L2-normalize embeddings (cho cosine similarity)
            batch_size: Batch size khi encode
            query_instruct: Instruction prefix cho queries
                           Ví dụ: "query: "
            passage_instruct: Instruction prefix cho passages
                             Ví dụ: "passage: "
            model_kwargs: Kwargs cho SentenceTransformer
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        
        # Setup cache directory
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_dpr_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        
        # Load SentenceTransformer model
        self.model = SentenceTransformer(
            model_name, 
            trust_remote_code=True, 
            model_kwargs=model_kwargs
        )
        
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

    def index(self, entity_list: list) -> None:
        """
        Index entities bằng cách encode thành embeddings.
        
        Workflow:
        1. Tạo fingerprint (MD5) từ entity_list
        2. Check cache: Nếu có → load embeddings
        3. Nếu không có cache:
           - Encode entities với SentenceTransformer
           - Lưu vào cache nếu use_cache=True
        
        Args:
            entity_list: Danh sách entities cần index
        
        Notes:
            - Cache file: {root}/{md5_hash}.pt
            - Embeddings shape: [num_entities, embedding_dim]
            - Dùng GPU nếu available
        """
        self.entity_list = entity_list
        
        # Tạo cache filename từ MD5 hash
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        
        # Load từ cache nếu có
        if os.path.exists(cache_file):
            self.entity_embeddings = torch.load(
                cache_file,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=True,
            )
        else:
            # Encode entities
            self.entity_embeddings = self.model.encode(
                entity_list,
                device="cuda" if torch.cuda.is_available() else "cpu",
                convert_to_tensor=True,
                show_progress_bar=True,
                prompt=self.passage_instruct,      # Thêm instruction prefix
                normalize_embeddings=self.normalize,  # L2 normalize
                batch_size=self.batch_size,
            )
            
            # Lưu cache
            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link NER entities với indexed KB entities.
        
        Workflow:
        1. Encode query entities
        2. Compute similarity: query_emb @ entity_emb.T
        3. Get top-k entities
        4. Normalize scores
        
        Args:
            ner_entity_list: Entities từ NER
            topk: Số candidates trả về
        
        Returns:
            dict: {query_entity: [candidates]}
            
        Notes:
            - Similarity = dot product (cosine nếu normalized)
            - norm_score = score / max_score (để so sánh relative)
        """
        # Encode query entities
        ner_entity_embeddings = self.model.encode(
            ner_entity_list,
            device="cuda" if torch.cuda.is_available() else "cpu",
            convert_to_tensor=True,
            prompt=self.query_instruct,           # Query instruction
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
        )
        
        # Compute similarity matrix: [num_queries, num_entities]
        scores = ner_entity_embeddings @ self.entity_embeddings.T
        
        # Get top-k
        top_k_scores, top_k_values = torch.topk(scores, topk, dim=-1)
        
        # Format results
        linked_entity_dict: dict[str, list] = {}
        for i in range(len(ner_entity_list)):
            linked_entity_dict[ner_entity_list[i]] = []

            sorted_score = top_k_scores[i]
            sorted_indices = top_k_values[i]
            max_score = sorted_score[0].item()  # Highest score để normalize

            # Thêm top-k candidates
            for score, top_k_index in zip(sorted_score, sorted_indices):
                linked_entity_dict[ner_entity_list[i]].append(
                    {
                        "entity": self.entity_list[top_k_index],
                        "score": score.item(),
                        "norm_score": score.item() / max_score,
                    }
                )
        
        return linked_entity_dict


class NVEmbedV2ELModel(DPRELModel):
    """
    Entity Linking model đặc biệt cho NVEmbed V2.
    
    NVEmbed V2 đặc điểm:
    - Hỗ trợ context length rất dài (32768 tokens)
    - Yêu cầu EOS token ở cuối input
    - Padding ở bên phải (right-side)
    
    Kế thừa DPRELModel và override:
    - __init__: Set max_seq_length=32768, padding_side="right"
    - __call__: Thêm EOS token vào queries trước khi encode
    
    Ví dụ:
        model = NVEmbedV2ELModel(
            'nvidia/NV-Embed-v2',
            query_instruct="Instruct: Retrieve equivalent entities\\nQuery: "
        )
        model.index(['Paris', 'London'])
        results = model(['paris city'], topk=2)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Khởi tạo với config đặc biệt cho NVEmbed V2.
        
        Modifications:
        - max_seq_length: 32768 (support long context)
        - padding_side: "right" (required by NVEmbed V2)
        """
        super().__init__(*args, **kwargs)
        
        # Override settings cho NVEmbed V2
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        """
        Thêm EOS token vào cuối mỗi input.
        
        Args:
            input_examples: List of text strings
        
        Returns:
            List of texts với EOS token appended
            
        Ví dụ:
            ["Paris", "London"] → ["Paris</s>", "London</s>"]
        """
        input_examples = [
            input_example + self.model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def __call__(self, ner_entity_list: list, *args: Any, **kwargs: Any) -> dict:
        """
        Link entities với EOS token preprocessing.
        
        Workflow:
        1. Thêm EOS token vào queries
        2. Gọi parent's __call__() để encode và search
        
        Args:
            ner_entity_list: Entities cần link
        
        Returns:
            dict: Same format như DPRELModel
        """
        # Thêm EOS token (required by NVEmbed V2)
        ner_entity_list = self.add_eos(ner_entity_list)
        
        # Gọi parent's __call__
        return super().__call__(ner_entity_list, *args, **kwargs)