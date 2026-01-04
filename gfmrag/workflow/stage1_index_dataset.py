"""
FILE: stage1_index_dataset.py
MÔ TẢ: Script này là giai đoạn 1 trong hệ thống RAG (Retrieval-Augmented Generation)
       Nhiệm vụ chính: Xây dựng Knowledge Graph (KG) từ dataset và tạo Q&A pairs
       
LUỒNG CHẠY TỔNG QUÁT:
1. Load configuration từ file config YAML
2. Khởi tạo KG Constructor (xây dựng Knowledge Graph)
3. Khởi tạo QA Constructor (tạo cặp câu hỏi-trả lời)
4. Sử dụng KGIndexer để index dữ liệu
"""

import logging
import os

import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gfmrag import KGIndexer
from gfmrag.kg_construction import KGConstructor, QAConstructor

# ============================================================================
# KHỞI TẠO LOGGER
# ============================================================================
# Logger dùng để ghi log thông tin trong quá trình chạy chương trình
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD BIẾN MÔI TRƯỜNG
# ============================================================================
# Load các biến môi trường từ file .env (ví dụ: API keys, database credentials)
dotenv.load_dotenv()


# ============================================================================
# HÀM MAIN - ĐIỂM VÀO CHÍNH CỦA CHƯƠNG TRÌNH
# ============================================================================
@hydra.main(config_path="config", config_name="stage1_index_dataset", version_base=None)
# Decorator @hydra.main: 
# - Tự động load config từ thư mục "config"
# - File config có tên "stage1_index_dataset.yaml"
# - Hydra giúp quản lý cấu hình linh hoạt, có thể override từ command line
def main(cfg: DictConfig) -> None:
    """
    Hàm main thực hiện toàn bộ pipeline index dataset
    
    THAM SỐ:
        cfg (DictConfig): Object chứa toàn bộ cấu hình từ file YAML
                          Bao gồm: kg_constructor config, qa_constructor config, dataset config
    
    LUỒNG THỰC THI:
        Bước 1: Lấy thông tin về output directory
        Bước 2: Log thông tin cấu hình và directories
        Bước 3: Khởi tạo KG Constructor từ config
        Bước 4: Khởi tạo QA Constructor từ config
        Bước 5: Tạo KGIndexer và index dữ liệu
    """
    
    # ========================================================================
    # BƯỚC 1: LẤY OUTPUT DIRECTORY
    # ========================================================================
    # HydraConfig.get() trả về cấu hình runtime của Hydra
    # output_dir là thư mục nơi Hydra sẽ lưu kết quả, logs
    # Mặc định: outputs/YYYY-MM-DD/HH-MM-SS/
    output_dir = HydraConfig.get().runtime.output_dir
    
    # ========================================================================
    # BƯỚC 2: LOG THÔNG TIN
    # ========================================================================
    # Log cấu hình dưới dạng YAML để dễ đọc và debug
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    
    # Log thư mục làm việc hiện tại
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Log thư mục output (nơi lưu kết quả)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # BƯỚC 3: KHỞI TẠO KG CONSTRUCTOR
    # ========================================================================
    # KGConstructor: Component chịu trách nhiệm xây dựng Knowledge Graph
    # - Trích xuất entities (thực thể) từ documents
    # - Xác định relationships (mối quan hệ) giữa các entities
    # - Tạo cấu trúc đồ thị kiến thức
    # from_config(): Factory method tạo object từ config dictionary
    kg_constructor = KGConstructor.from_config(cfg.kg_constructor)
    
    # ========================================================================
    # BƯỚC 4: KHỞI TẠO QA CONSTRUCTOR (OPTIONAL)
    # ========================================================================
    # QAConstructor: Component tạo các cặp Question-Answer từ documents
    # - Sinh câu hỏi tự động từ nội dung
    # - Tạo câu trả lời tương ứng
    # - Dùng để augment (tăng cường) dữ liệu training hoặc evaluation
    # NOTE: QA constructor is now optional - if not in config, it will be None
    qa_constructor = None
    if "qa_constructor" in cfg:
        logger.info("QA constructor found in config - initializing...")
        qa_constructor = QAConstructor.from_config(cfg.qa_constructor)
    else:
        logger.info("QA constructor not in config - skipping QA functionality")

    # ========================================================================
    # BƯỚC 5: INDEX DỮ LIỆU
    # ========================================================================
    # KGIndexer: Component chính kết hợp KG và QA construction
    # - Nhận vào kg_constructor và qa_constructor
    # - Điều phối toàn bộ quá trình index
    kg_indexer = KGIndexer(kg_constructor, qa_constructor)
    
    # index_data(): Hàm thực hiện index dataset
    # INPUT: cfg.dataset - chứa thông tin về dataset cần index
    #        (đường dẫn, format, các tham số liên quan)
    # OUTPUT: Knowledge Graph + Q&A pairs được lưu vào storage/database
    # 
    # QUÁ TRÌNH BÊN TRONG index_data():
    # 1. Load dataset từ cfg.dataset
    # 2. Với mỗi document:
    #    a. Dùng kg_constructor để extract entities và relationships
    #    b. Dùng qa_constructor để generate Q&A pairs
    # 3. Lưu trữ Knowledge Graph vào vector database/graph database
    # 4. Lưu trữ Q&A pairs để sử dụng sau
    kg_indexer.index_data(cfg.dataset)


# ============================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ============================================================================
if __name__ == "__main__":
    """
    Entry point khi chạy script trực tiếp
    
    CÁCH CHẠY:
        python stage1_index_dataset.py
        
    HOẶC OVERRIDE CONFIG:
        python stage1_index_dataset.py dataset.path=/path/to/data
        python stage1_index_dataset.py kg_constructor.model=gpt-4
    
    Hydra sẽ tự động:
    - Parse command line arguments
    - Load config từ file YAML
    - Merge với overrides từ command line
    - Gọi hàm main(cfg)
    """
    main()


"""
===============================================================================
TÓM TẮT LUỒNG CHẠY
===============================================================================

START
  │
  ├─► Load .env file (API keys, credentials)
  │
  ├─► @hydra.main decorator
  │   └─► Load config từ config/stage1_index_dataset.yaml
  │
  ├─► main(cfg) được gọi
  │
  ├─► Lấy output_dir từ Hydra
  │
  ├─► Log thông tin config và directories
  │
  ├─► Khởi tạo KGConstructor
  │   └─► Chuẩn bị component xây dựng Knowledge Graph
  │
  ├─► Khởi tạo QAConstructor  
  │   └─► Chuẩn bị component tạo Q&A pairs
  │
  ├─► Tạo KGIndexer với 2 constructors
  │
  ├─► kg_indexer.index_data(dataset)
  │   │
  │   ├─► Load dataset
  │   │
  │   ├─► For each document:
  │   │   ├─► Extract entities & relationships → Knowledge Graph
  │   │   └─► Generate Q&A pairs
  │   │
  │   └─► Save to storage (vector DB, graph DB)
  │
END

===============================================================================
CÁC COMPONENT CHÍNH
===============================================================================

1. KGConstructor:
   - Xây dựng Knowledge Graph từ text
   - Extract entities (người, địa điểm, tổ chức, concepts...)
   - Identify relationships giữa entities
   
2. QAConstructor:
   - Tự động sinh câu hỏi từ documents
   - Tạo câu trả lời tương ứng
   - Augment training/evaluation data
   
3. KGIndexer:
   - Orchestrator điều phối toàn bộ pipeline
   - Kết hợp KG construction và QA generation
   - Quản lý việc lưu trữ kết quả

===============================================================================
MỤC ĐÍCH CỦA STAGE 1
===============================================================================

Stage 1 là bước chuẩn bị dữ liệu cho hệ thống RAG:
- Chuyển đổi raw documents → structured knowledge (Knowledge Graph)
- Tạo Q&A pairs để hỗ trợ retrieval sau này
- Index và lưu trữ để stage 2 có thể query/retrieve thông tin

Sau stage 1, hệ thống có:
✓ Knowledge Graph với entities và relationships
✓ Q&A pairs để training/evaluation
✓ Indexed data sẵn sàng cho retrieval

===============================================================================
"""