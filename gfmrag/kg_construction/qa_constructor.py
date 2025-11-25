"""
================================================================================
FILE: qa_constructor.py - Xây dựng Dataset Question-Answering
================================================================================

MÔ TẢ TỔNG QUAN:
File này định nghĩa các class để xây dựng dataset Question-Answering từ dữ liệu
thô. Quy trình xử lý bao gồm:
1. Named Entity Recognition (NER) trên các câu hỏi
2. Entity Linking (EL) để liên kết entities với Knowledge Graph
3. Kết hợp entities từ câu hỏi với entities từ supporting documents

KIẾN TRÚC:
- BaseQAConstructor: Abstract base class định nghĩa interface
- QAConstructor: Implementation cụ thể với NER và Entity Linking

CHỨC NĂNG CHÍNH:
1. NER trên câu hỏi để tìm các entities
2. Entity Linking để chuẩn hóa entities với KG
3. Trích xuất supporting entities từ documents
4. Tạo final dataset với đầy đủ thông tin entities

QUY TRÌNH XỬ LÝ:
Input: Raw QA data (questions + supporting_facts)
  ↓
1. NER: Nhận dạng entities trong questions
  ↓
2. Entity Linking: Liên kết với entities trong KG
  ↓
3. Extract supporting entities từ documents
  ↓
Output: Processed QA data (questions + question_entities + supporting_entities)

DEPENDENCIES:
- NER Model: Để nhận dạng entities trong text
- Entity Linking Model: Để map entities với KG
- Knowledge Graph: Đã được xây dựng trước bởi KGConstructor

FILE OUTPUTS:
- ner_results.jsonl: Kết quả NER (cached để tái sử dụng)
- Processed QA data với entities đã được link

VÍ DỤ SỬ DỤNG:
    qa_constructor = QAConstructor.from_config(config)
    processed_data = qa_constructor.prepare_data(
        data_root="./data",
        data_name="my_dataset",
        file="train.json"
    )
    
CẤU TRÚC DỮ LIỆU:
Input sample:
    {
        "id": "q_123",
        "question": "Who founded Microsoft?",
        "supporting_facts": ["doc1", "doc2"]
    }

Output sample:
    {
        "id": "q_123",
        "question": "Who founded Microsoft?",
        "supporting_facts": ["doc1", "doc2"],
        "question_entities": ["Microsoft"],  # Từ NER + EL
        "supporting_entities": ["Bill Gates", "Microsoft", "1975"]  # Từ docs
    }
================================================================================
"""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag.kg_construction.utils import KG_DELIMITER

from .entity_linking_model import BaseELModel
from .ner_model import BaseNERModel

logger = logging.getLogger(__name__)


class BaseQAConstructor(ABC):
    """
    Abstract base class (lớp cơ sở trừu tượng) cho việc xây dựng QA datasets.
    
    Đây là interface định nghĩa các phương thức bắt buộc mà các subclass phải implement.
    Sử dụng Abstract Base Class (ABC) để đảm bảo tính nhất quán trong thiết kế.
    
    Thuộc tính:
        Không có thuộc tính cụ thể
    
    Phương thức:
        prepare_data: Phương thức trừu tượng bắt buộc phải được implement
                      để chuẩn bị dữ liệu QA
    """

    @abstractmethod
    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Phương thức trừu tượng để chuẩn bị dữ liệu QA.
        
        Subclass phải override phương thức này để implement logic xử lý cụ thể.
        
        Tham số:
            data_root (str): Đường dẫn đến thư mục gốc chứa dataset
            data_name (str): Tên của dataset
            file (str): Tên file cần xử lý (ví dụ: "train.json", "test.json")
        
        Returns:
            list[dict]: Danh sách các samples đã được xử lý
                       Mỗi dict chứa thông tin câu hỏi và entities liên quan
        """
        pass


class QAConstructor(BaseQAConstructor):
    """
    Class xây dựng QA dataset với NER và Entity Linking.
    
    Class này xử lý raw QA datasets bằng cách:
    1. Thực hiện Named Entity Recognition (NER) trên các câu hỏi
    2. Thực hiện Entity Linking (EL) để kết nối entities với Knowledge Graph
    3. Trích xuất supporting entities từ documents
    
    Tham số:
        ner_model (BaseNERModel): Mô hình NER để nhận dạng entities
        el_model (BaseELModel): Mô hình Entity Linking để link entities với KG
        root (str): Thư mục gốc cho các file tạm (default: "tmp/qa_construction")
        num_processes (int): Số processes cho xử lý song song (default: 1)
        force (bool): Có bắt buộc tính toán lại kết quả đã cache không (default: False)
    
    Thuộc tính:
        ner_model: Instance của NER model
        el_model: Instance của Entity Linking model
        root: Đường dẫn thư mục gốc
        num_processes: Số processes song song
        data_name: Tên dataset đang được xử lý
        force: Cờ để force recompute
        DELIMITER: Ký tự phân cách trong KG files
    
    Phương thức:
        from_config: Tạo QAConstructor từ configuration
        prepare_data: Xử lý raw QA data để thêm thông tin entities
    
    Lưu ý:
        - Cần có Knowledge Graph và document2entities mapping đã được tạo trước
        - Các file này phải nằm trong thư mục processed/stage1 của dataset
        - NER results được cache trong file ner_results.jsonl để tái sử dụng
    """

    DELIMITER = KG_DELIMITER  # Ký tự phân cách trong KG (thường là "\t" hoặc " ")

    def __init__(
        self,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        root: str = "tmp/qa_construction",
        num_processes: int = 1,
        force: bool = False,
    ) -> None:
        """
        Khởi tạo Question Answer Constructor.
        
        Constructor này thiết lập các thành phần cần thiết để xử lý text data
        thông qua NER và Entity Linking models.
        
        Tham số:
            ner_model (BaseNERModel): Mô hình để Named Entity Recognition
            el_model (BaseELModel): Mô hình để Entity Linking
            root (str): Thư mục gốc để lưu dữ liệu đã xử lý (default: "tmp/qa_construction")
            num_processes (int): Số processes cho xử lý song song (default: 1)
                                 num_processes > 1 cho phép xử lý nhiều câu hỏi cùng lúc
            force (bool): Nếu True, bắt buộc xử lý lại dữ liệu đã tồn tại (default: False)
        
        Thuộc tính được khởi tạo:
            ner_model: Instance NER model đã được khởi tạo
            el_model: Instance EL model đã được khởi tạo
            root: Đường dẫn thư mục gốc
            num_processes: Số processes song song
            data_name: Tên dataset (khởi tạo là None, sẽ được set khi xử lý)
            force: Cờ để force reprocessing
        """
        self.ner_model = ner_model
        self.el_model = el_model
        self.root = root
        self.num_processes = num_processes
        self.data_name = None  # Sẽ được set khi gọi prepare_data
        self.force = force

    @property
    def tmp_dir(self) -> str:
        """
        Trả về đường dẫn thư mục tạm để xử lý dữ liệu.
        
        Property này tạo và trả về đường dẫn thư mục đặc thù cho data_name hiện tại
        dưới thư mục root. Thư mục sẽ được tạo nếu chưa tồn tại.
        
        Mục đích: Tách biệt dữ liệu tạm của các datasets khác nhau
        
        Returns:
            str: Đường dẫn đến thư mục tạm
                 Format: {root}/{data_name}/
        
        Raises:
            AssertionError: Nếu data_name chưa được set trước khi gọi property này
        
        Ví dụ:
            Nếu root="tmp/qa" và data_name="hotpotqa"
            Trả về: "tmp/qa/hotpotqa/"
        """
        # Kiểm tra data_name đã được set chưa
        assert (
            self.data_name is not None
        ), "data_name must be set before accessing tmp_dir"
        
        # Tạo đường dẫn thư mục tạm
        tmp_dir = os.path.join(self.root, self.data_name)
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        return tmp_dir

    @staticmethod
    def from_config(cfg: DictConfig) -> "QAConstructor":
        """
        Tạo QAConstructor instance từ configuration.
        
        Phương thức static này khởi tạo QAConstructor sử dụng các tham số từ config,
        tạo một thư mục tạm duy nhất dựa trên fingerprint của config để lưu trữ
        các artifacts xử lý.
        
        Quy trình:
        1. Chuyển config thành dict và xóa tham số 'force' (không ảnh hưởng fingerprint)
        2. Tạo fingerprint (MD5 hash) từ config
        3. Tạo thư mục tạm với tên là fingerprint
        4. Lưu config vào file config.json trong thư mục đó
        5. Khởi tạo và trả về QAConstructor với các models từ config
        
        Tham số:
            cfg (DictConfig): Configuration object chứa:
                - root: Thư mục gốc
                - ner_model: Config cho NER model
                - el_model: Config cho Entity Linking model
                - num_processes: Số processes xử lý song song
                - force: Cờ force reprocessing (optional)
        
        Returns:
            QAConstructor: Instance đã được khởi tạo với config đã cho
        
        Lưu ý:
            - Fingerprint giúp phân biệt các lần chạy với config khác nhau
            - Config được lưu lại để truy xuất sau này
            - Sử dụng Hydra's instantiate để tạo models từ config
        """
        # Bước 1: Chuyển config thành dictionary và loại bỏ 'force'
        config = OmegaConf.to_container(cfg, resolve=True)
        if "force" in config:
            del config["force"]  # Không tính 'force' vào fingerprint
        
        # Bước 2: Tạo fingerprint từ config
        # MD5 hash đảm bảo cùng config -> cùng fingerprint
        fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

        # Bước 3: Tạo thư mục với tên là fingerprint
        base_tmp_dir = os.path.join(cfg.root, fingerprint)
        if not os.path.exists(base_tmp_dir):
            os.makedirs(base_tmp_dir)
            
            # Bước 4: Lưu config vào file để tham khảo
            json.dump(
                config,
                open(os.path.join(base_tmp_dir, "config.json"), "w"),
                indent=4,
            )
        
        # Bước 5: Khởi tạo QAConstructor
        return QAConstructor(
            root=base_tmp_dir,
            ner_model=instantiate(cfg.ner_model),      # Tạo NER model từ config
            el_model=instantiate(cfg.el_model),        # Tạo EL model từ config
            num_processes=cfg.num_processes,
            force=cfg.force,
        )

    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Chuẩn bị dữ liệu cho question answering bằng cách xử lý raw data,
        thực hiện Named Entity Recognition (NER) và Entity Linking (EL).
        
        QUY TRÌNH XỬ LÝ CHI TIẾT:
        
        1. SETUP VÀ LOAD DỮ LIỆU:
           - Set data_name và xác định các đường dẫn
           - Xóa cache nếu force=True
           - Load Knowledge Graph để lấy danh sách entities
           - Load document2entities mapping
           - Load raw QA data
        
        2. NER - NAMED ENTITY RECOGNITION:
           - Kiểm tra cache: Load kết quả NER đã có (nếu có)
           - Xác định samples chưa được xử lý
           - Xử lý song song với ThreadPool:
             * Đối với mỗi câu hỏi: gọi NER model
             * Nhận dạng các entities trong câu hỏi
           - Lưu kết quả vào ner_results.jsonl (cache)
        
        3. ENTITY LINKING:
           - Index tất cả entities từ KG vào EL model
           - Thu thập tất cả entities đã nhận dạng từ NER
           - Gọi EL model để link entities với KG
           - Mỗi NER entity -> entity chuẩn trong KG (topk=1: lấy match tốt nhất)
        
        4. EXTRACT SUPPORTING ENTITIES:
           - Đối với mỗi sample:
             * Lấy supporting_facts (documents liên quan)
             * Trích xuất entities từ các documents này (dùng doc2entities)
             * Tạo danh sách supporting_entities
        
        5. TẠO FINAL DATA:
           - Kết hợp:
             * Dữ liệu gốc (id, question, supporting_facts)
             * question_entities (từ NER + EL)
             * supporting_entities (từ documents)
           - Mỗi sample có đầy đủ thông tin để train/evaluate
        
        Tham số:
            data_root (str): Thư mục gốc chứa dataset
                            Ví dụ: "./data"
            data_name (str): Tên dataset
                            Ví dụ: "hotpotqa"
            file (str): Tên file raw data cần xử lý
                       Ví dụ: "train.json", "test.json"
        
        Returns:
            list[dict]: Danh sách các samples đã xử lý. Mỗi sample là dict chứa:
                - Các fields gốc từ raw data
                - question_entities (list): Entities được link từ câu hỏi
                - supporting_entities (list): Entities từ supporting documents
        
        Raises:
            FileNotFoundError: Nếu file KG không tồn tại trong thư mục processed
        
        Files được tạo/sử dụng:
            - INPUT:
              * {data_root}/{data_name}/raw/{file}: Raw QA data
              * {data_root}/{data_name}/processed/stage1/kg.txt: Knowledge Graph
              * {data_root}/{data_name}/processed/stage1/document2entities.json: Doc mapping
            
            - OUTPUT (cache):
              * {tmp_dir}/ner_results.jsonl: Kết quả NER (để tái sử dụng)
        
        Ví dụ:
            constructor = QAConstructor(ner_model, el_model)
            data = constructor.prepare_data(
                data_root="./data",
                data_name="hotpotqa", 
                file="train.json"
            )
            
            # data[0] có thể là:
            {
                "id": "q_123",
                "question": "Who founded Microsoft?",
                "supporting_facts": ["doc1", "doc2"],
                "question_entities": ["Microsoft"],
                "supporting_entities": ["Bill Gates", "Microsoft", "1975"]
            }
        """
        # ====================================================================
        # BƯỚC 1: SETUP VÀ LOAD DỮ LIỆU
        # ====================================================================
        
        # Set tên dataset (cần cho tmp_dir property)
        self.data_name = data_name  # type: ignore
        
        # Xác định đường dẫn
        raw_path = os.path.join(data_root, data_name, "raw", file)
        processed_path = os.path.join(data_root, data_name, "processed", "stage1")

        # Nếu force=True, xóa cache để xử lý lại từ đầu
        if self.force:
            logger.info("Force mode: Clearing cache in tmp directory")
            for tmp_file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, tmp_file))

        # Kiểm tra KG file có tồn tại không (bắt buộc)
        if not os.path.exists(os.path.join(processed_path, "kg.txt")):
            raise FileNotFoundError(
                "KG file not found. Please run KG construction first"
            )

        # Load Knowledge Graph để lấy danh sách entities
        logger.info("Loading Knowledge Graph")
        entities = set()
        with open(os.path.join(processed_path, "kg.txt")) as f:
            for line in f:
                try:
                    # Phân tách mỗi triple: head, relation, tail
                    u, _, v = line.strip().split(self.DELIMITER)
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                # Thu thập tất cả entities (cả head và tail)
                entities.add(u)
                entities.add(v)
        
        # Load document-to-entities mapping
        logger.info("Loading document2entities mapping")
        with open(os.path.join(processed_path, "document2entities.json")) as f:
            doc2entities = json.load(f)

        # Load raw QA data
        logger.info(f"Loading raw data from {raw_path}")
        with open(raw_path) as f:
            data = json.load(f)

        # ====================================================================
        # BƯỚC 2: NER - NAMED ENTITY RECOGNITION
        # ====================================================================
        
        ner_results = {}  # Dictionary để lưu kết quả NER: {id: result}
        
        # Thử load kết quả NER đã có từ cache
        ner_cache_path = os.path.join(self.tmp_dir, "ner_results.jsonl")
        if os.path.exists(ner_cache_path):
            logger.info("Loading cached NER results")
            with open(ner_cache_path) as f:
                ner_logs = [json.loads(line) for line in f]
                ner_results = {log["id"]: log for log in ner_logs}

        # Xác định samples chưa được xử lý
        unprocessed_data = [
            sample for sample in data if sample["id"] not in ner_results
        ]
        
        logger.info(f"Processing NER for {len(unprocessed_data)} unprocessed samples")

        def _ner_process(sample: dict) -> dict:
            """
            Hàm helper để xử lý NER cho một sample.
            
            Được gọi trong ThreadPool để xử lý song song nhiều samples.
            
            Tham số:
                sample (dict): Sample chứa 'id' và 'question'
            
            Returns:
                dict: Kết quả NER chứa:
                    - id: ID của sample
                    - question: Câu hỏi gốc
                    - ner_ents: Danh sách entities được nhận dạng
            """
            id = sample["id"]
            question = sample["question"]
            
            # Gọi NER model để nhận dạng entities trong câu hỏi
            ner_ents = self.ner_model(question)
            
            return {
                "id": id,
                "question": question,
                "ner_ents": ner_ents,  # Danh sách entity strings
            }

        # Xử lý NER song song với ThreadPool
        if unprocessed_data:
            with ThreadPool(self.num_processes) as pool:
                # Mở file để append kết quả NER
                with open(ner_cache_path, "a") as f:
                    # imap: ánh xạ _ner_process lên mỗi sample, xử lý song song
                    for res in tqdm(
                        pool.imap(_ner_process, unprocessed_data),
                        total=len(unprocessed_data),
                        desc="NER Processing"
                    ):
                        # Lưu kết quả vào dict
                        ner_results[res["id"]] = res
                        # Ghi vào file cache (mỗi dòng một JSON)
                        f.write(json.dumps(res) + "\n")

        # ====================================================================
        # BƯỚC 3: ENTITY LINKING
        # ====================================================================
        
        logger.info("Performing Entity Linking")
        
        # Index tất cả entities từ KG vào EL model
        # Điều này cho phép EL model tìm kiếm nhanh entities tương tự
        self.el_model.index(list(entities))

        # Thu thập tất cả entities đã nhận dạng từ NER
        ner_entities = []
        for res in ner_results.values():
            ner_entities.extend(res["ner_ents"])

        # Gọi EL model để link mỗi NER entity với entity chuẩn trong KG
        # topk=1: Chỉ lấy match tốt nhất cho mỗi entity
        # Kết quả: {ner_entity: [{"entity": kg_entity, "score": similarity_score}]}
        el_results = self.el_model(ner_entities, topk=1)

        # ====================================================================
        # BƯỚC 4 & 5: EXTRACT SUPPORTING ENTITIES VÀ TẠO FINAL DATA
        # ====================================================================
        
        logger.info("Creating final dataset")
        final_data = []
        
        for sample in data:
            id = sample["id"]
            
            # Lấy kết quả NER cho sample này
            ner_ents = ner_results[id]["ner_ents"]
            
            # Link mỗi NER entity với entity chuẩn trong KG
            question_entities = []
            for ent in ner_ents:
                # Lấy entity chuẩn (match tốt nhất từ EL)
                question_entities.append(el_results[ent][0]["entity"])

            # Trích xuất supporting entities từ documents
            supporting_facts = sample.get("supporting_facts", [])
            supporting_entities = []
            
            # Đối với mỗi supporting document
            for item in list(set(supporting_facts)):  # set() để loại duplicate
                # Thêm tất cả entities từ document đó
                supporting_entities.extend(doc2entities.get(item, []))

            # Tạo sample hoàn chỉnh
            final_data.append(
                {
                    **sample,  # Giữ nguyên các fields gốc
                    "question_entities": question_entities,      # Entities từ câu hỏi
                    "supporting_entities": supporting_entities,  # Entities từ docs
                }
            )

        logger.info(f"Processed {len(final_data)} samples")
        return final_data