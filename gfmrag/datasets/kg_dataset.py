"""
================================================================================
FILE: kg_dataset.py - Dataset cho Knowledge Graph (Đồ thị tri thức)
================================================================================

MÔ TÃ TỔNG QUAN:
File này định nghĩa class KGDataset để xử lý và quản lý dữ liệu Knowledge Graph.
Knowledge Graph là một cấu trúc dữ liệu đại diện cho các thực thể và mối quan hệ 
giữa chúng dưới dạng bộ ba (entity-relation-entity triplets).

CHỨC NĂNG CHÍNH:
1. Đọc và xử lý file dữ liệu KG từ định dạng text (kg.txt)
2. Chuyển đổi các bộ ba (head, relation, tail) thành cấu trúc đồ thị PyTorch
3. Tạo các quan hệ nghịch đảo (inverse relations) để hỗ trợ học hai chiều
4. Sinh embedding (vector đặc trưng) cho các quan hệ bằng mô hình embedding văn bản
5. Lưu trữ và quản lý ánh xạ giữa tên thực thể/quan hệ và ID số
6. Cung cấp interface để truy xuất dữ liệu đã xử lý

CẤU TRÚC DỮ LIỆU ĐẦU VÀO:
- kg.txt: File chứa các bộ ba dạng "entity1 [delimiter] relation [delimiter] entity2"

CẤU TRÚC DỮ LIỆU ĐẦU RA:
- data.pt: File chứa đồ thị đã xử lý
- ent2id.json: Ánh xạ từ tên thực thể sang ID
- rel2id.json: Ánh xạ từ tên quan hệ sang ID (bao gồm cả quan hệ nghịch đảo)
- text_emb_model_cfgs.json: Cấu hình mô hình embedding

VÍ DỤ SỬ DỤNG:
    kg_dataset = KGDataset(
        root="./data",
        data_name="my_kg",
        text_emb_model_cfgs=config,
        force_rebuild=False
    )
================================================================================
"""

import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.kg_construction.utils import KG_DELIMITER
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank

logger = logging.getLogger(__name__)


class KGDataset(InMemoryDataset):
    """
    Class Dataset cho Knowledge Graph kế thừa từ InMemoryDataset của PyTorch Geometric.
    
    Class này xử lý dữ liệu đồ thị tri thức, bao gồm các bộ ba entity-relation-entity,
    và hỗ trợ xử lý cả quan hệ thuận và nghịch.
    
    Tham số:
        root (str): Thư mục gốc để lưu dataset
        data_name (str): Tên dataset
        text_emb_model_cfgs (DictConfig): Cấu hình mô hình embedding văn bản
        force_rebuild (bool): Có bắt buộc build lại dữ liệu đã xử lý không
        **kwargs: Các tham số bổ sung
    
    Thuộc tính:
        name (str): Tên của dataset
        fingerprint (str): Mã hash MD5 của cấu hình mô hình embedding
        delimiter (str): Ký tự phân cách trong file KG text
        data (Data): Đối tượng dữ liệu đồ thị đã xử lý
        slices (dict): Các slice dữ liệu cho batching
    """

    # Ký tự phân cách giữa các phần tử trong bộ ba (head, relation, tail)
    delimiter = KG_DELIMITER

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
        **kwargs: str,
    ) -> None:
        """
        Khởi tạo KGDataset.
        
        Quy trình:
        1. Lưu tên dataset và cờ force_rebuild
        2. Tạo fingerprint từ config để phân biệt các phiên bản xử lý khác nhau
        3. Gọi constructor của class cha (InMemoryDataset)
        4. Load dữ liệu đã xử lý từ file
        5. Lấy số chiều của relation embedding
        """
        self.name = data_name
        self.force_rebuild = force_rebuild
        
        # Tạo fingerprint (dấu vân tay) từ config của mô hình embedding
        # Mục đích: Phân biệt các lần xử lý khác nhau khi thay đổi config
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        
        self.text_emb_model_cfgs = text_emb_model_cfgs
        
        # Gọi constructor của class cha để khởi tạo dataset
        super().__init__(root, None, None)
        
        # Load dữ liệu đồ thị đã được xử lý và lưu trước đó
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        # Lấy số chiều của relation embedding (ví dụ: 768 cho BERT)
        self.feat_dim = self._data.rel_emb.size(1)

    @property
    def raw_file_names(self) -> list:
        """
        Trả về danh sách tên các file raw cần thiết.
        
        Returns:
            list: Danh sách chứa "kg.txt" - file chứa các bộ ba Knowledge Graph
        """
        return ["kg.txt"]

    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        """
        Đọc file Knowledge Graph và xử lý thành dữ liệu có cấu trúc.
        
        Chức năng:
        - Đọc từng dòng trong file kg.txt
        - Phân tách mỗi dòng thành bộ ba (head, relation, tail)
        - Tạo ID duy nhất cho mỗi entity và relation
        - Xây dựng vocabulary mapping (tên -> ID)
        
        Tham số:
            triplet_file (str): Đường dẫn đến file chứa các bộ ba
            inv_entity_vocab (dict): Dictionary ánh xạ tên entity -> ID (sẽ được cập nhật)
            inv_rel_vocab (dict): Dictionary ánh xạ tên relation -> ID (sẽ được cập nhật)
        
        Returns:
            dict: Dictionary chứa:
                - triplets: Danh sách các bộ ba đã chuyển thành ID
                - num_node: Tổng số nodes (entities)
                - num_relation: Tổng số relations
                - inv_entity_vocab: Vocabulary của entities
                - inv_rel_vocab: Vocabulary của relations
        """
        triplets = []  # Danh sách lưu các bộ ba với ID số
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # Đọc file và xử lý từng dòng
        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    # Phân tách dòng thành 3 phần: head entity, relation, tail entity
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                
                # Nếu entity chưa có trong vocabulary, thêm vào và gán ID mới
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                
                # Nếu relation chưa có trong vocabulary, thêm vào và gán ID mới
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                
                # Chuyển đổi từ tên sang ID
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                # Lưu bộ ba dưới dạng (head_id, tail_id, relation_id)
                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

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
            logger.warning(f"Processing KG dataset {self.name} at rank {get_rank()}")
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
        Xử lý dataset Knowledge Graph từ dữ liệu thô.
        
        Quy trình xử lý bao gồm:
        1. Load file kg.txt và tạo vocabulary cho entities và relations
        2. Chuyển đổi các bộ ba thành tensor PyTorch
        3. Tạo các cạnh nghịch đảo (inverse edges) để mô hình có thể học hai chiều
        4. Lưu mapping ent2id và rel2id ra file JSON
        5. Sinh relation embeddings sử dụng mô hình text embedding
        6. Tạo đối tượng Data của PyTorch Geometric
        7. Lưu toàn bộ dữ liệu đã xử lý
        
        Dữ liệu đầu ra:
        - edge_index: Tensor chứa các cạnh (bao gồm cả cạnh nghịch đảo)
        - edge_type: Loại quan hệ của mỗi cạnh
        - target_edge_index: Các cạnh gốc (không bao gồm nghịch đảo)
        - target_edge_type: Loại quan hệ của các cạnh gốc
        - num_nodes: Tổng số nodes
        - num_relations: Tổng số relations (x2 vì có nghịch đảo)
        - rel_emb: Relation embeddings
        """
        kg_file = self.raw_paths[0]

        # Bước 1: Load file và tạo vocabulary
        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        # Lấy số lượng nodes và relations
        # Lưu ý: Một số dataset có entities mới xuất hiện trong test set
        num_node = kg_result["num_node"]
        num_relations = kg_result["num_relation"]

        kg_triplets = kg_result["triplets"]

        # Bước 2: Chuyển các bộ ba thành tensor cho các cạnh gốc
        # train_target_edges shape: [2, num_edges] với row 0 là source, row 1 là target
        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        # Bước 3: Tạo cạnh nghịch đảo
        # Ví dụ: (A, relation, B) -> thêm (B, inverse_relation, A)
        # flip(0) đảo ngược source và target
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        # Cạnh nghịch đảo có ID = ID_gốc + num_relations
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        # Bước 4: Lưu entity vocabulary ra file JSON
        with open(self.processed_dir + "/ent2id.json", "w") as f:
            json.dump(kg_result["inv_entity_vocab"], f)
        
        # Bước 5: Tạo và lưu relation vocabulary (bao gồm cả inverse relations)
        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}  # Đảo ngược mapping
        
        # Thêm tên cho các quan hệ nghịch đảo
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()
        
        with open(self.processed_dir + "/rel2id.json", "w") as f:
            json.dump(rel2id, f)

        # Bước 6: Sinh relation embeddings
        logger.info("Generating relation embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        # Encode tất cả tên relations (kể cả inverse) thành vectors
        rel_emb = text_emb_model.encode(list(rel2id.keys()), is_query=False).cpu()

        # Bước 7: Tạo đối tượng Data của PyTorch Geometric
        kg_data = Data(
            edge_index=train_edges,  # Tất cả cạnh (gốc + nghịch đảo)
            edge_type=train_etypes,  # Loại của tất cả cạnh
            num_nodes=num_node,  # Tổng số nodes
            target_edge_index=train_target_edges,  # Chỉ cạnh gốc (để đánh giá)
            target_edge_type=train_target_etypes,  # Loại của cạnh gốc
            num_relations=num_relations * 2,  # x2 vì có nghịch đảo
            rel_emb=rel_emb,  # Relation embeddings
        )

        # Bước 8: Lưu dữ liệu đã xử lý
        torch.save((self.collate([kg_data])), self.processed_paths[0])

        # Lưu cấu hình mô hình embedding để tham khảo sau này
        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)

    def __repr__(self) -> str:
        """
        Trả về chuỗi đại diện cho dataset.
        
        Returns:
            str: Tên dataset dạng "DatasetName()"
        """
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        """
        Trả về tổng số loại relations trong dataset.
        
        Returns:
            int: Số lượng relation types (bao gồm cả nghịch đảo)
        """
        return int(self.data.edge_type.max()) + 1

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
        
        Sử dụng fingerprint để phân biệt các phiên bản xử lý khác nhau
        dựa trên config của mô hình embedding.
        
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
        Trả về tên file chứa dữ liệu đã xử lý.
        
        Returns:
            str: "data.pt" - file chứa đồ thị đã xử lý
        """
        return "data.pt"