"""
================================================================================
FILE: qa_dataset.py - Dataset cho Question Answering trên Knowledge Graph
================================================================================

MÔ TẢ TỔNG QUAN:
File này định nghĩa class QADataset để xử lý và quản lý dữ liệu cho bài toán
Question Answering (Hỏi đáp) dựa trên Knowledge Graph. Dataset này được xây dựng
trên nền tảng KGDataset và xử lý các câu hỏi cùng với câu trả lời dựa trên
thông tin trong đồ thị tri thức.

CHỨC NĂNG CHÍNH:
1. Xử lý dữ liệu câu hỏi từ file JSON (train.json, test.json)
2. Sinh embedding (vector đặc trưng) cho các câu hỏi
3. Tạo các mask (mặt nạ) để đánh dấu:
   - Entities xuất hiện trong câu hỏi
   - Entities hỗ trợ trả lời (supporting entities)
   - Documents chứa thông tin liên quan (supporting documents)
4. Kết nối dữ liệu QA với Knowledge Graph
5. Tạo mapping giữa entities và documents chứa chúng
6. Cung cấp interface để train và evaluate mô hình QA

CẤU TRÚC DỮ LIỆU ĐẦU VÀO:
- train.json / test.json: File chứa các câu hỏi, entities liên quan, documents hỗ trợ
- dataset_corpus.json: Tập hợp các documents
- document2entities.json: Mapping từ document tới các entities trong đó
- kg.txt: File Knowledge Graph (được xử lý bởi KGDataset)

CẤU TRÚC DỮ LIỆU ĐẦU RA:
- qa_data.pt: File chứa dữ liệu QA đã xử lý
- ent2doc.pt: Sparse tensor ánh xạ từ entity sang document
- text_emb_model_cfgs.json: Cấu hình mô hình embedding

ĐỊNH DẠNG MỖI MẪU DỮ LIỆU:
{
    "question": "What is...?",
    "question_entities": ["entity1", "entity2"],
    "supporting_entities": ["entity3", "entity4"],
    "supporting_facts": ["doc1", "doc2"]
}

VÍ DỤ SỬ DỤNG:
    qa_dataset = QADataset(
        root="./data",
        data_name="my_qa",
        text_emb_model_cfgs=config,
        force_rebuild=False
    )
    train_data = qa_dataset.data[0]  # Training split
    test_data = qa_dataset.data[1]   # Test split
================================================================================
"""

import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import datasets
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class QADataset(InMemoryDataset):
    """
    Class Dataset cho bài toán Question-Answering dựa trên Knowledge Graph.
    
    Class này kế thừa từ InMemoryDataset của PyTorch Geometric và xử lý dữ liệu
    câu hỏi-đáp thành định dạng phù hợp cho các mô hình graph-based QA.
    Nó xử lý cả tập training và test.
    
    Tham số:
        root (str): Thư mục gốc để lưu dataset
        data_name (str): Tên dataset
        text_emb_model_cfgs (DictConfig): Cấu hình mô hình embedding cho câu hỏi
        force_rebuild (bool): Có bắt buộc xử lý lại dataset không
    
    Thuộc tính chính:
        name (str): Tên dataset
        kg (KGDataset): Dataset Knowledge Graph nền tảng
        rel_emb_dim (int): Số chiều của relation embeddings
        ent2id (dict): Mapping từ tên entity sang ID
        rel2id (dict): Mapping từ tên relation sang ID
        doc (dict): Corpus các documents
        doc2entities (dict): Mapping từ document sang các entities trong đó
        raw_train_data (list): Dữ liệu training thô
        raw_test_data (list): Dữ liệu test thô
        ent2docs (torch.Tensor): Sparse tensor ánh xạ entity->documents
        id2doc (dict): Mapping từ ID document sang tên document
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
    ):
        """
        Khởi tạo QADataset.
        
        Quy trình:
        1. Lưu các tham số cơ bản (name, force_rebuild, config)
        2. Tạo fingerprint từ config mô hình embedding
        3. Khởi tạo KGDataset để load Knowledge Graph
        4. Gọi constructor của class cha
        5. Load dữ liệu QA đã xử lý
        6. Load các properties bổ sung (ent2id, rel2id, documents, etc.)
        """
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.text_emb_model_cfgs = text_emb_model_cfgs
        
        # Tạo fingerprint để phân biệt các phiên bản xử lý khác nhau
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        
        # Khởi tạo Knowledge Graph dataset
        # KG là nền tảng cho việc trả lời câu hỏi
        kg = KGDataset(root, data_name, text_emb_model_cfgs, force_rebuild)
        self.kg = kg[0]  # Lấy graph đầu tiên từ KGDataset
        self.feat_dim = kg.feat_dim  # Số chiều của feature embeddings
        
        # Gọi constructor của class cha
        super().__init__(root, None, None)
        
        # Load dữ liệu QA đã được xử lý và lưu trước đó
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        
        # Load các thuộc tính bổ sung (mappings, corpus, etc.)
        self.load_property()

    def __repr__(self) -> str:
        """
        Trả về chuỗi đại diện cho dataset.
        
        Returns:
            str: Tên dataset dạng "DatasetName()"
        """
        return f"{self.name}()"

    @property
    def raw_file_names(self) -> list:
        """
        Trả về danh sách tên các file raw cần thiết.
        
        Returns:
            list: ["train.json", "test.json"] - các file chứa dữ liệu QA
        """
        return ["train.json", "test.json"]

    @property
    def raw_dir(self) -> str:
        """
        Trả về đường dẫn đến thư mục chứa dữ liệu thô (stage1).
        
        Returns:
            str: Đường dẫn tuyệt đối đến thư mục raw
        """
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        """
        Trả về đường dẫn đến thư mục chứa dữ liệu đã xử lý (stage2).
        
        Sử dụng fingerprint để phân biệt các phiên bản xử lý khác nhau.
        
        Returns:
            str: Đường dẫn tuyệt đối đến thư mục processed
        """
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        """
        Trả về tên file chứa dữ liệu QA đã xử lý.
        
        Returns:
            str: "qa_data.pt" - file chứa dữ liệu QA
        """
        return "qa_data.pt"

    def load_property(self) -> None:
        """
        Load các thuộc tính cần thiết từ files đã được xử lý.
        
        Chức năng:
        - Load entity và relation vocabularies (ent2id, rel2id)
        - Load corpus các documents
        - Load mapping từ document sang entities
        - Load dữ liệu train và test thô (nếu có)
        - Load sparse tensor ánh xạ entity sang documents
        - Tạo mapping từ ID document sang tên document
        
        Các thuộc tính được load:
        - self.ent2id: {entity_name: entity_id}
        - self.rel2id: {relation_name: relation_id}
        - self.doc: {doc_name: doc_content}
        - self.doc2entities: {doc_name: [entity_names]}
        - self.raw_train_data: List các samples training
        - self.raw_test_data: List các samples test
        - self.ent2docs: Sparse tensor shape (n_entities, n_docs)
        - self.id2doc: {doc_id: doc_name}
        """
        # Load entity vocabulary
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        
        # Load relation vocabulary
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        
        # Load corpus - tập hợp các documents
        with open(
            os.path.join(str(self.root), str(self.name), "raw", "dataset_corpus.json")
        ) as fin:
            self.doc = json.load(fin)
        
        # Load mapping từ document sang các entities xuất hiện trong document
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)
        
        # Load dữ liệu training (nếu file tồn tại)
        if os.path.exists(os.path.join(self.raw_dir, "train.json")):
            with open(os.path.join(self.raw_dir, "train.json")) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.raw_train_data = []
        
        # Load dữ liệu test (nếu file tồn tại)
        if os.path.exists(os.path.join(self.raw_dir, "test.json")):
            with open(os.path.join(self.raw_dir, "test.json")) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.raw_test_data = []

        # Load sparse tensor ánh xạ từ entity sang documents chứa entity đó
        # Shape: (n_entities, n_documents)
        # ent2docs[i, j] = 1 nếu entity i xuất hiện trong document j
        self.ent2docs = torch.load(
            os.path.join(self.processed_dir, "ent2doc.pt"), weights_only=True
        )
        
        # Tạo mapping từ document ID sang tên document
        self.id2doc = {i: doc for i, doc in enumerate(self.doc2entities)}

    def _process(self) -> None:
        """
        Phương thức nội bộ kiểm tra và trigger quá trình xử lý dataset.
        
        Chức năng:
        - Kiểm tra xem dữ liệu đã được xử lý chưa
        - Kiểm tra tính hợp lệ của pre_transform và pre_filter
        - Gọi phương thức process() nếu cần xử lý lại
        - Tạo thư mục processed_dir nếu chưa tồn tại
        """
        # Kiểm tra pre_transform có thay đổi không
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        # Kiểm tra pre_filter có thay đổi không
        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        # Nếu force_rebuild=True hoặc file chưa tồn tại, tiến hành xử lý
        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning(f"Processing QA dataset {self.name} at rank {get_rank()}")
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            # Tạo thư mục processed nếu chưa có
            makedirs(self.processed_dir)
            self.process()

            # Lưu thông tin pre_transform và pre_filter
            path = osp.join(self.processed_dir, "pre_transform.pt")
            torch.save(_repr(self.pre_transform), path)
            path = osp.join(self.processed_dir, "pre_filter.pt")
            torch.save(_repr(self.pre_filter), path)

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        """
        Xử lý và chuẩn bị dataset cho bài toán Question-Answering.
        
        Quy trình xử lý gồm các bước chính:
        
        1. LOAD MAPPINGS:
           - Load ent2id: ánh xạ entity name -> entity ID
           - Load rel2id: ánh xạ relation name -> relation ID
           - Load doc2entities: ánh xạ document -> các entities trong đó
        
        2. TẠO ENTITY-DOCUMENT MAPPING:
           - Chuyển đổi doc2entities thành ent2doc (sparse tensor)
           - Mục đích: Nhanh chóng tìm documents chứa một entity cụ thể
        
        3. XỬ LÝ CÁC SAMPLES:
           Với mỗi sample (train hoặc test):
           a) Lấy question và các entities liên quan
           b) Chuyển entity names thành entity IDs
           c) Tạo masks:
              - question_entities_mask: đánh dấu entities trong câu hỏi
              - supporting_entities_mask: đánh dấu entities hỗ trợ trả lời
              - supporting_docs_mask: đánh dấu documents chứa thông tin liên quan
           d) Skip sample nếu thiếu thông tin cần thiết
        
        4. SINH QUESTION EMBEDDINGS:
           - Sử dụng text embedding model để encode tất cả câu hỏi
           - Chuyển thành vector đặc trưng để mô hình có thể xử lý
        
        5. TẠO DATASET:
           - Đóng gói tất cả dữ liệu đã xử lý
           - Chia thành các splits (train, test)
           - Lưu vào file qa_data.pt
        
        6. LƯU CẤU HÌNH:
           - Lưu config của text embedding model để tham khảo
        
        Files được tạo:
        - ent2doc.pt: Sparse tensor (n_entities x n_docs)
        - qa_data.pt: Processed QA dataset với các splits
        - text_emb_model_cfgs.json: Cấu hình mô hình embedding
        
        Cấu trúc mỗi sample trong dataset:
        - question_embeddings: Vector embedding của câu hỏi
        - question_entities_masks: Binary mask cho entities trong câu hỏi
        - supporting_entities_masks: Binary mask cho supporting entities
        - supporting_docs_masks: Binary mask cho supporting documents
        - sample_id: ID của sample trong file gốc
        """
        # Bước 1: Load các mappings cần thiết
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        with open(os.path.join(self.raw_dir, "document2entities.json")) as fin:
            self.doc2entities = json.load(fin)

        # Lấy số lượng nodes từ Knowledge Graph
        num_nodes = self.kg.num_nodes
        
        # Tạo mapping từ document name sang document ID
        doc2id = {doc: i for i, doc in enumerate(self.doc2entities)}
        
        # Bước 2: Tạo entity-to-document mapping
        n_docs = len(self.doc2entities)
        
        # Tạo ma trận doc2ent: doc2ent[i,j]=1 nếu entity j xuất hiện trong doc i
        doc2ent = torch.zeros((n_docs, num_nodes))
        for doc, entities in self.doc2entities.items():
            # Chuyển entity names thành IDs (chỉ lấy entities có trong vocabulary)
            entity_ids = [self.ent2id[ent] for ent in entities if ent in self.ent2id]
            doc2ent[doc2id[doc], entity_ids] = 1
        
        # Chuyển vị thành ent2doc và chuyển thành sparse tensor để tiết kiệm bộ nhớ
        ent2doc = doc2ent.T.to_sparse()  # Shape: (n_entities, n_docs)
        torch.save(ent2doc, os.path.join(self.processed_dir, "ent2doc.pt"))

        # Bước 3: Khởi tạo các danh sách để lưu dữ liệu đã xử lý
        sample_id = []  # ID của mỗi sample
        questions = []  # Danh sách các câu hỏi (text)
        question_entities_masks = []  # Masks cho entities trong câu hỏi
        supporting_entities_masks = []  # Masks cho entities hỗ trợ
        supporting_docs_masks = []  # Masks cho documents hỗ trợ
        num_samples = []  # Số lượng samples trong mỗi split

        # Xử lý từng file (train.json, test.json)
        for path in self.raw_paths:
            if not os.path.exists(path):
                num_samples.append(0)
                continue  # Bỏ qua nếu file không tồn tại
            
            num_sample = 0
            with open(path) as fin:
                data = json.load(fin)
                
                # Xử lý từng sample trong file
                for index, item in enumerate(data):
                    # Chuyển entity names trong câu hỏi thành IDs
                    question_entities = [
                        self.ent2id[x]
                        for x in item["question_entities"]
                        if x in self.ent2id
                    ]

                    # Chuyển supporting entity names thành IDs
                    supporting_entities = [
                        self.ent2id[x]
                        for x in item["supporting_entities"]
                        if x in self.ent2id
                    ]

                    # Chuyển supporting document names thành IDs
                    supporting_docs = [
                        doc2id[doc] for doc in item["supporting_facts"] if doc in doc2id
                    ]

                    # Bỏ qua sample nếu bất kỳ thông tin nào bị thiếu
                    # Điều này đảm bảo chất lượng dữ liệu
                    if any(
                        len(x) == 0
                        for x in [
                            question_entities,
                            supporting_entities,
                            supporting_docs,
                        ]
                    ):
                        continue
                    
                    # Sample hợp lệ, thêm vào dataset
                    num_sample += 1
                    sample_id.append(index)
                    question = item["question"]
                    questions.append(question)

                    # Tạo binary masks từ danh sách IDs
                    # Mask có shape (num_nodes,) với giá trị 1 tại vị trí entities
                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )

                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )

                    # Tương tự cho documents
                    supporting_docs_masks.append(
                        entities_to_mask(supporting_docs, n_docs)
                    )
                
                # Lưu số lượng samples đã xử lý cho split này
                num_samples.append(num_sample)

        # Bước 4: Sinh question embeddings
        logger.info("Generating question embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        
        # Encode tất cả câu hỏi thành vectors
        # is_query=True cho biết đây là query (câu hỏi) chứ không phải document
        question_embeddings = text_emb_model.encode(
            questions,
            is_query=True,
        ).cpu()
        
        # Chuyển các danh sách masks thành tensors
        question_entities_masks = torch.stack(question_entities_masks)
        supporting_entities_masks = torch.stack(supporting_entities_masks)
        supporting_docs_masks = torch.stack(supporting_docs_masks)
        sample_id = torch.tensor(sample_id, dtype=torch.long)

        # Bước 5: Tạo HuggingFace Dataset từ dữ liệu đã xử lý
        dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "question_entities_masks": question_entities_masks,
                "supporting_entities_masks": supporting_entities_masks,
                "supporting_docs_masks": supporting_docs_masks,
                "sample_id": sample_id,
            }
        ).with_format("torch")  # Đảm bảo format tương thích với PyTorch
        
        # Chia dataset thành các splits (train, test)
        offset = 0
        splits = []
        for num_sample in num_samples:
            # Tạo subset cho mỗi split
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        
        # Lưu tất cả splits vào file
        torch.save(splits, self.processed_paths[0])

        # Bước 6: Lưu cấu hình mô hình embedding
        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)