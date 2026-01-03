"""
colbert_el_model.py - Entity Linking dùng ColBERT

ColBERT: Neural IR model với late interaction mechanism
- Encode entities thành token-level embeddings
- Search bằng MaxSim operation
"""

import hashlib
import logging
import os
import shutil

from ragatouille import RAGPretrainedModel

from gfmrag.kg_construction.utils import processing_phrases

from .base_model import BaseELModel

logger = logging.getLogger(__name__)


class ColbertELModel(BaseELModel):
    """
    Entity Linking model dùng ColBERT (Contextualized Late Interaction over BERT).
    
    ColBERT đặc điểm:
    - Late interaction: Tính similarity ở token level, sau đó aggregate
    - Token-level embeddings: Mỗi token có embedding riêng
    - MaxSim: Similarity = sum of max similarities cho mỗi query token
    
    Workflow:
    1. index(): Encode và lưu entity embeddings
    2. __call__(): Query entities, tìm top-k matches bằng MaxSim
    
    Ví dụ:
        model = ColbertELModel("colbert-ir/colbertv2.0")
        model.index(["Paris", "London", "Berlin"])
        
        results = model(["paris city"], topk=2)
        # → {'paris city': [
        #       {'entity': 'Paris', 'score': 0.82, 'norm_score': 1.0},
        #       {'entity': 'London', 'score': 0.35, 'norm_score': 0.43}
        #    ]}
    """

    def __init__(
        self,
        model_name_or_path: str = "colbert-ir/colbertv2.0",
        root: str = "tmp",
        force: bool = False,
        **kwargs: str,
    ) -> None:
        """
        Khởi tạo ColBERT model.
        
        Args:
            model_name_or_path: Path hoặc HF model name
                               Default: "colbert-ir/colbertv2.0"
            root: Thư mục lưu index
            force: Xóa index cũ và tạo lại nếu True
        """
        self.model_name_or_path = model_name_or_path
        self.root = root
        self.force = force
        
        # Load pretrained ColBERT qua RAGatouille wrapper
        self.colbert_model = RAGPretrainedModel.from_pretrained(
            self.model_name_or_path,
            index_root=self.root,
        )

    def index(self, entity_list: list) -> None:
        """
        Index entities với ColBERT.
        
        Workflow:
        1. Tạo fingerprint (MD5 hash) từ entity_list
        2. Check cache: Nếu index đã tồn tại và force=False → reuse
        3. Clean entities bằng processing_phrases()
        4. Encode và lưu vào FAISS index
        
        Args:
            entity_list: Danh sách entities cần index
        
        Notes:
            - Index name format: "Entity_index_{md5_hash}"
            - Dùng FAISS cho fast similarity search
            - split_documents=False: Mỗi entity là 1 document nguyên vẹn
        """
        # Tạo unique index name từ MD5 hash
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        index_name = f"Entity_index_{fingerprint}"
        
        # Xóa index cũ nếu force=True
        if os.path.exists(f"{self.root}/colbert/{fingerprint}") and self.force:
            shutil.rmtree(f"{self.root}/colbert/{fingerprint}")
        
        # Clean entities: lowercase, remove special chars
        phrases = [processing_phrases(p) for p in entity_list]
        
        # Tạo ColBERT index
        index_path = self.colbert_model.index(
            index_name=index_name,
            collection=phrases,
            overwrite_index=self.force if self.force else "reuse",
            split_documents=False,  # Không split entities thành chunks
            use_faiss=True,         # Dùng FAISS cho speed
        )
        self.index_path = index_path

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link NER entities với indexed KB entities.
        
        Workflow:
        1. Clean query entities bằng processing_phrases()
        2. Search trong ColBERT index với MaxSim
        3. Normalize scores (chia cho max_score)
        4. Return top-k candidates cho mỗi query
        
        Args:
            ner_entity_list: Entities từ NER cần link
            topk: Số candidates trả về
        
        Returns:
            dict: {query_entity: [candidates]}
            
        Raises:
            AttributeError: Nếu chưa gọi index() trước
        
        Notes:
            - norm_score = score / max_score trong batch
            - Max_score để tránh division by zero = 1.0 nếu không có results
        """
        # Check xem đã index chưa
        try:
            self.__getattribute__("index_path")
        except AttributeError as e:
            raise AttributeError("Index the entities first using index method") from e

        # Clean query entities
        queries = [processing_phrases(p) for p in ner_entity_list]

        # Search với ColBERT
        results = self.colbert_model.search(queries, k=topk)

        # Debug: Log raw result structure
        logger.debug(f"ColBERT search returned results type: {type(results)}")
        if results and len(results) > 0:
            logger.debug(f"First result type: {type(results[0])}, Sample: {results[0][:2] if results[0] else 'Empty'}")

        # Format kết quả
        linked_entity_dict: dict[str, list] = {}
        for i in range(len(queries)):
            query = queries[i]
            result = results[i]
            linked_entity_dict[query] = []

            # Validate result format
            if not result:
                logger.debug(f"Empty result for query '{query}'")
                continue

            # Debug: Check raw result structure
            logger.debug(f"Query '{query}' got {len(result)} results")
            if result:
                logger.debug(f"First result type: {type(result[0])}, content: {str(result[0])[:100]}")

            # Check if results are in expected format
            valid_results = []
            for r in result:
                # Handle different result formats from RAGatouille
                if isinstance(r, dict):
                    # Expected format: dict with 'content' and 'score' keys
                    if "content" in r and "score" in r:
                        valid_results.append(r)
                    # Alternative format: 'text' instead of 'content'
                    elif "text" in r and "score" in r:
                        valid_results.append({
                            "content": r["text"],
                            "score": r["score"]
                        })
                    else:
                        logger.warning(f"Unexpected result format for query '{query}': keys={list(r.keys())}")
                elif isinstance(r, str):
                    # If result is just a string, skip it with warning
                    logger.warning(f"Got string result instead of dict for query '{query}': '{r[:50]}...'")
                else:
                    logger.warning(f"Unexpected result type for query '{query}': {type(r)}")

            if not valid_results:
                logger.warning(f"No valid results for query '{query}' after format validation. Total results: {len(result)}")
                continue

            # Tính max_score để normalize (default 1.0 tránh division by zero)
            max_score = max([r["score"] for r in valid_results])

            # Thêm candidates với normalized scores
            for r in valid_results:
                linked_entity_dict[query].append(
                    {
                        "entity": r["content"],          # Entity từ KB
                        "score": r["score"],             # Raw ColBERT score
                        "norm_score": r["score"] / max_score,  # Normalized
                    }
                )

        return linked_entity_dict

    def compute_pairwise_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute pairwise similarity between two entities using ColBERT.

        This method indexes entity2 and searches for entity1, returning the similarity score.
        Useful for evaluating entity resolution quality.

        Args:
            entity1: First entity string
            entity2: Second entity string

        Returns:
            float: Similarity score (0-1), or 0.0 if computation fails

        Example:
            >>> model = ColbertELModel()
            >>> model.compute_pairwise_similarity("aspirin", "acetylsalicylic acid")
            0.856
        """
        try:
            # Index the second entity
            self.index([entity2])

            # Search for the first entity
            result_dict = self([entity1], topk=1)

            # Extract score
            cleaned_entity1 = processing_phrases(entity1)
            if cleaned_entity1 in result_dict and result_dict[cleaned_entity1]:
                # Return the raw score (not normalized)
                return result_dict[cleaned_entity1][0].get("score", 0.0)
            else:
                logger.warning(f"No results found for pairwise similarity: '{entity1}' vs '{entity2}'")
                return 0.0

        except Exception as e:
            logger.error(f"ColBERT pairwise similarity failed for '{entity1}' and '{entity2}': {e}")
            return 0.0