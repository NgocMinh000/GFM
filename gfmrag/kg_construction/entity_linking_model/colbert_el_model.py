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


def _setup_hf_token():
    """Load HF token from .env and set as environment variable."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # Set token for transformers library
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            logger.info("HF token loaded from .env")
            return True
        return False
    except ImportError:
        logger.debug("python-dotenv not installed, skipping token setup")
        return False


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
                               Nếu local model tồn tại tại models/colbert/colbertv2.0,
                               sẽ tự động dùng local model đó.
            root: Thư mục lưu index
            force: Xóa index cũ và tạo lại nếu True
        """
        # Setup HF token from .env if available (cho private models hoặc rate limiting)
        _setup_hf_token()

        # Ưu tiên dùng local model nếu có (tránh download từ HuggingFace)
        local_model_path = "models/colbert/colbertv2.0"
        if os.path.exists(local_model_path) and model_name_or_path == "colbert-ir/colbertv2.0":
            logger.info(f"Using local ColBERT model from: {local_model_path}")
            self.model_name_or_path = local_model_path
        else:
            self.model_name_or_path = model_name_or_path
            if not os.path.exists(model_name_or_path):
                logger.warning(
                    f"Local model not found at {local_model_path}. "
                    f"Will download from HuggingFace: {model_name_or_path}"
                )

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
            try:
                sample = str(results[0])[:100] if results[0] else 'Empty'
            except Exception:
                sample = f"<{type(results[0]).__name__}>"
            logger.debug(f"First result type: {type(results[0])}, Sample: {sample}")

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
                try:
                    content_preview = str(result[0])[:100]
                except Exception:
                    content_preview = f"<{type(result[0]).__name__}>"
                logger.debug(f"First result type: {type(result[0])}, content: {content_preview}")

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
                    try:
                        preview = r[:50] if len(r) > 50 else r
                    except Exception:
                        preview = "<unable to preview>"
                    logger.warning(f"Got string result instead of dict for query '{query}': '{preview}...'")
                else:
                    # Handle any other unexpected types
                    try:
                        type_info = f"{type(r).__name__}: {str(r)[:50]}"
                    except Exception:
                        type_info = f"{type(r).__name__}"
                    logger.warning(f"Unexpected result type for query '{query}': {type_info}")

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
        Compute pairwise similarity between two entities using ColBERT MaxSim.

        FIXED: Index both entities together WITHOUT FAISS clustering.
        This avoids FAISS clustering errors when dealing with small entity sets.

        Algorithm:
        1. Index both entities together (use_faiss=False to avoid clustering)
        2. Search entity1 → find entity2's score
        3. Search entity2 → find entity1's score
        4. Average bidirectional scores for symmetry

        Args:
            entity1: First entity string
            entity2: Second entity string

        Returns:
            float: Similarity score [0.0, 1.0], or 0.0 if computation fails

        Example:
            >>> model = ColbertELModel()
            >>> model.compute_pairwise_similarity("aspirin", "acetylsalicylic acid")
            0.856

        Note:
            - Uses temporary index without FAISS
            - Bidirectional similarity for symmetry
            - Safe for single entity pairs
        """
        try:
            # Clean entities
            entity1_clean = processing_phrases(entity1)
            entity2_clean = processing_phrases(entity2)

            # Create temporary index name
            temp_fingerprint = hashlib.md5(f"{entity1}{entity2}".encode()).hexdigest()
            temp_index_name = f"Pairwise_{temp_fingerprint}"

            # Index both entities together WITHOUT FAISS
            # This avoids clustering errors for small entity sets
            self.colbert_model.index(
                index_name=temp_index_name,
                collection=[entity1_clean, entity2_clean],
                overwrite_index=True,
                split_documents=False,
                use_faiss=False,  # CRITICAL: Disable FAISS to avoid clustering
            )

            # Search entity1 → find entity2
            results_1to2 = self.colbert_model.search(query=entity1_clean, k=2)
            score_1to2 = 0.0
            if results_1to2 and len(results_1to2) > 0:
                # Find entity2 in results
                for result in results_1to2:
                    if isinstance(result, dict):
                        content = result.get("content") or result.get("text", "")
                        if content == entity2_clean:
                            score_1to2 = result.get("score", 0.0)
                            break

            # Search entity2 → find entity1
            results_2to1 = self.colbert_model.search(query=entity2_clean, k=2)
            score_2to1 = 0.0
            if results_2to1 and len(results_2to1) > 0:
                # Find entity1 in results
                for result in results_2to1:
                    if isinstance(result, dict):
                        content = result.get("content") or result.get("text", "")
                        if content == entity1_clean:
                            score_2to1 = result.get("score", 0.0)
                            break

            # Bidirectional average for symmetry
            final_score = (score_1to2 + score_2to1) / 2.0

            # Normalize to [0, 1] range (ColBERT scores can vary)
            final_score = max(0.0, min(1.0, final_score))

            logger.debug(f"ColBERT similarity '{entity1}' ↔ '{entity2}': {final_score:.4f} "
                       f"(1→2: {score_1to2:.4f}, 2→1: {score_2to1:.4f})")

            # Cleanup: Remove temporary index
            temp_index_path = f"{self.root}/colbert/indexes/{temp_index_name}"
            if os.path.exists(temp_index_path):
                shutil.rmtree(temp_index_path)

            return final_score

        except Exception as e:
            logger.error(f"ColBERT pairwise similarity failed for '{entity1}' and '{entity2}': {e}")
            import traceback
            traceback.print_exc()
            return 0.0