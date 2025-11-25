"""
FILE: stage2_kg_pretrain.py
MÔ TẢ: Script này là giai đoạn 2 - Pre-training mô hình trên Knowledge Graph
       Nhiệm vụ chính: Huấn luyện mô hình để học biểu diễn (representation) của 
                      entities và relations trong Knowledge Graph
       
LUỒNG CHẠY TỔNG QUÁT:
1. Khởi tạo distributed training environment (nếu dùng nhiều GPU)
2. Load datasets từ Knowledge Graph đã được index ở stage 1
3. Khởi tạo model và optimizer
4. Train model qua nhiều epochs với task Link Prediction
5. Validate và save best model
6. Test trên test set và save pretrained model

MỤC TIÊU: 
- Pre-train model để học embeddings tốt cho entities và relations
- Model này sẽ được dùng cho các tasks downstream (QA, retrieval...)
"""

import logging
import math
import os
from itertools import islice

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from torch_geometric.data import Data
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import KGDataset
from gfmrag.ultra import tasks
from gfmrag.utils import GraphDatasetLoader

# ============================================================================
# KHỞI TẠO LOGGER VÀ SEPARATORS
# ============================================================================
logger = logging.getLogger(__name__)

# Các ký tự separator để format log cho dễ đọc
separator = ">" * 30  # Ngăn cách các phần lớn
line = "-" * 30       # Ngăn cách các phần nhỏ


# ============================================================================
# HÀM 1: CREATE_KGC_DATASET - TẠO DATASET CHO KNOWLEDGE GRAPH COMPLETION
# ============================================================================
def create_kgc_dataset(
    dataset: dict[str, KGDataset],
    batch_size: int,
    world_size: int,
    rank: int,
    is_train: bool = True,
    shuffle: bool = True,
    fast_test: None | int = None,
) -> dict:
    """
    Tạo dataset và dataloader cho task Knowledge Graph Completion (KGC)
    
    THAM SỐ:
        dataset: Dictionary chứa KGDataset với keys:
                 - "data_name": tên dataset
                 - "data": list chứa graph data
        batch_size: Số lượng triples trong mỗi batch
        world_size: Tổng số GPUs/processes trong distributed training
        rank: ID của process hiện tại (0 đến world_size-1)
        is_train: True nếu đang training, False nếu testing
        shuffle: Có shuffle data không
        fast_test: Nếu không None, chỉ lấy N triples đầu tiên để test nhanh
    
    RETURN:
        Dictionary chứa:
        - data_name: tên dataset
        - graph: toàn bộ graph
        - val_filtered_data: data để evaluation (filtered setting)
        - data_loader: PyTorch DataLoader
    
    KIẾN THỨC:
        Triple trong KG: (head, relation, tail) hoặc (h, r, t)
        Ví dụ: (Barack_Obama, president_of, USA)
    """
    
    # ========================================================================
    # BƯỚC 1: EXTRACT DỮ LIỆU TỪ DATASET
    # ========================================================================
    data_name = dataset["data_name"]  # Tên dataset (ví dụ: "FB15k-237")
    graph = dataset["data"][0]        # Graph đầu tiên trong list
    
    # ========================================================================
    # BƯỚC 2: TẠO FILTERED DATA CHO EVALUATION
    # ========================================================================
    # Filtered setting: Khi đánh giá model, loại bỏ các triples đúng khác
    # để tránh penalty không công bằng
    # 
    # val_filtered_data chứa TẤT CẢ các triples đúng trong dataset
    # Dùng để filter ra các negative samples không hợp lệ khi evaluate
    val_filtered_data = Data(
        edge_index=graph.target_edge_index,  # [2, num_edges]: [heads, tails]
        edge_type=graph.target_edge_type,    # [num_edges]: relation types
        num_nodes=graph.num_nodes,           # Tổng số nodes (entities)
    )
    
    # ========================================================================
    # BƯỚC 3: TẠO TRIPLES TENSOR
    # ========================================================================
    # Triple = (head, tail, relation_type)
    # Chuyển từ format graph sang format triples để dễ train
    
    if not is_train and fast_test is not None:
        # ====================================================================
        # CHẾ ĐỘ FAST TEST: Chỉ lấy một phần nhỏ data để test nhanh
        # ====================================================================
        # Random permutation để lấy ngẫu nhiên fast_test triples
        mask = torch.randperm(graph.target_edge_index.shape[1])[:fast_test]
        
        # Lấy subset của edges
        sampled_target_edge_index = graph.target_edge_index[:, mask]  # [2, fast_test]
        sampled_target_edge_type = graph.target_edge_type[mask]       # [fast_test]
        
        # Tạo triples tensor: [num_triples, 3] với 3 cột là [head, tail, rel]
        triples = torch.cat(
            [sampled_target_edge_index, sampled_target_edge_type.unsqueeze(0)]
        ).t()
    else:
        # ====================================================================
        # CHẾ ĐỘ BÌNH THƯỜNG: Dùng toàn bộ data
        # ====================================================================
        # target_edge_index: [2, num_edges] 
        #   - Row 0: head entities
        #   - Row 1: tail entities
        # target_edge_type: [num_edges] - relation type cho mỗi edge
        
        # Concatenate và transpose để có shape [num_edges, 3]
        # Mỗi row là một triple: [head_id, tail_id, relation_type]
        triples = torch.cat(
            [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]
        ).t()
    
    # ========================================================================
    # BƯỚC 4: TẠO DISTRIBUTED SAMPLER
    # ========================================================================
    # DistributedSampler: Chia đều data cho các GPUs khác nhau
    # - Mỗi GPU chỉ xử lý một phần data
    # - Đảm bảo không có overlap giữa các GPUs
    # - shuffle=True để randomize thứ tự
    sampler = torch_data.DistributedSampler(
        triples, 
        num_replicas=world_size,  # Tổng số GPUs
        rank=rank,                 # ID của GPU hiện tại
        shuffle=shuffle
    )
    
    # ========================================================================
    # BƯỚC 5: TẠO DATALOADER
    # ========================================================================
    # DataLoader: Tạo batches từ triples
    # - Mỗi batch có batch_size triples
    # - Sampler đảm bảo distributed training đúng cách
    data_loader = torch_data.DataLoader(
        triples,
        batch_size=batch_size,
        sampler=sampler,
    )
    
    # ========================================================================
    # BƯỚC 6: TẠO LẠI VAL_FILTERED_DATA (có vẻ trùng lặp với bước 2?)
    # ========================================================================
    # Note: Đoạn này trùng với code ở trên, có thể là do refactoring
    val_filtered_data = Data(
        edge_index=graph.target_edge_index,
        edge_type=graph.target_edge_type,
        num_nodes=graph.num_nodes,
    )
    
    # ========================================================================
    # RETURN DICTIONARY
    # ========================================================================
    return {
        "data_name": data_name,              # Tên dataset
        "graph": graph,                      # Toàn bộ graph (để model dùng)
        "val_filtered_data": val_filtered_data,  # Data để filtered evaluation
        "data_loader": data_loader,          # DataLoader để iterate batches
    }


# ============================================================================
# HÀM 2: TRAIN_AND_VALIDATE - VÒNG LẶP HUẤN LUYỆN VÀ VALIDATION
# ============================================================================
def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    dataset_loader: GraphDatasetLoader,
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> None:
    """
    Hàm chính để train model qua nhiều epochs và validate
    
    THAM SỐ:
        cfg: Configuration từ Hydra
        output_dir: Thư mục lưu checkpoints
        model: Model cần train
        dataset_loader: Loader để load nhiều datasets
        device: CPU hoặc CUDA device
        batch_per_epoch: Giới hạn số batches mỗi epoch (None = không giới hạn)
    
    TASK: Link Prediction / Knowledge Graph Completion
    - Cho trước (h, r, ?): Dự đoán tail entity
    - Cho trước (?, r, t): Dự đoán head entity
    
    TRAINING STRATEGY:
    1. Positive triple: (h, r, t) thật từ KG
    2. Negative sampling: Tạo các triples sai (h, r, t') hoặc (h', r, t)
    3. Model phải score positive cao hơn negatives
    """
    
    # ========================================================================
    # EARLY RETURN NẾU KHÔNG CẦN TRAIN
    # ========================================================================
    if cfg.train.num_epoch == 0:
        return
    
    # ========================================================================
    # SETUP DISTRIBUTED TRAINING
    # ========================================================================
    world_size = utils.get_world_size()  # Tổng số GPUs
    rank = utils.get_rank()               # ID của GPU hiện tại
    
    # ========================================================================
    # BƯỚC 1: KHỞI TẠO OPTIMIZER
    # ========================================================================
    # instantiate: Hydra helper để tạo object từ config
    # Ví dụ config: optimizer: {_target_: torch.optim.Adam, lr: 0.001}
    optimizer = instantiate(cfg.optimizer, model.parameters())
    
    start_epoch = 0  # Epoch bắt đầu (có thể resume từ checkpoint)
    
    # ========================================================================
    # BƯỚC 2: LOAD CHECKPOINT NẾU TỒN TẠI (RESUME TRAINING)
    # ========================================================================
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            # Load checkpoint từ file
            state = torch.load(
                cfg.train.checkpoint, 
                map_location="cpu",     # Load lên CPU trước
                weights_only=True       # Security: chỉ load weights
            )
            
            # Restore optimizer state (learning rate, momentum, etc.)
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            else:
                logger.warning(
                    f"Optimizer state not found in {cfg.train.checkpoint}, "
                    "using default optimizer."
                )
            
            # Restore epoch để tiếp tục từ đúng chỗ
            if "epoch" in state:
                start_epoch = state["epoch"]
                logger.warning(f"Resuming training from epoch {start_epoch}.")
        else:
            logger.warning(
                f"Checkpoint {cfg.train.checkpoint} does not exist, "
                "using default optimizer."
            )
    
    # ========================================================================
    # BƯỚC 3: WRAP MODEL CHO DISTRIBUTED TRAINING
    # ========================================================================
    if world_size > 1:
        # DistributedDataParallel: PyTorch's multi-GPU training
        # - Synchronize gradients giữa các GPUs
        # - Mỗi GPU train trên một phần data khác nhau
        # - Aggregate gradients và update model đồng bộ
        parallel_model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device]
        )
    else:
        # Single GPU: không cần wrap
        parallel_model = model
    
    # ========================================================================
    # KHỞI TẠO BIẾN TRACKING
    # ========================================================================
    best_result = float("-inf")  # Best validation score (càng cao càng tốt)
    best_epoch = -1              # Epoch nào đạt best score
    batch_id = 0                 # Global batch counter (qua các epochs)
    
    # ========================================================================
    # VÒNG LẶP QUA CÁC EPOCHS
    # ========================================================================
    for i in range(start_epoch, cfg.train.num_epoch):
        epoch = i + 1  # Đánh số epoch từ 1
        parallel_model.train()  # Set model về training mode
        
        # ====================================================================
        # LOG BẮT ĐẦU EPOCH (chỉ rank 0)
        # ====================================================================
        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} begin")
        
        losses = []  # Lưu loss của tất cả batches trong epoch
        
        # ====================================================================
        # SET EPOCH CHO DATASET LOADER
        # ====================================================================
        # Đảm bảo thứ tự datasets giống nhau trên tất cả GPUs
        # Quan trọng cho reproducibility trong distributed training
        dataset_loader.set_epoch(epoch)
        
        # ====================================================================
        # VÒNG LẶP QUA TỪNG DATASET
        # ====================================================================
        # dataset_loader có thể load nhiều KG datasets khác nhau
        # Train trên tất cả datasets trong một epoch
        for train_dataset in dataset_loader:
            # ================================================================
            # CHUẨN BỊ DATASET
            # ================================================================
            train_dataset = create_kgc_dataset(
                train_dataset,
                cfg.train.batch_size,
                world_size,
                rank,
                is_train=True,
                shuffle=True,
            )
            
            data_name = train_dataset["data_name"]
            train_loader = train_dataset["data_loader"]
            
            # Set epoch cho sampler để shuffle đúng cách
            train_loader.sampler.set_epoch(epoch)
            
            # Move graph lên GPU
            train_graph = train_dataset["graph"].to(device)
            
            # ================================================================
            # VÒNG LẶP QUA TỪNG BATCH
            # ================================================================
            # islice: Giới hạn số batches nếu batch_per_epoch được set
            # tqdm: Progress bar để theo dõi tiến độ
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {data_name}: {epoch}",
                total=batch_per_epoch,
            ):
                # ============================================================
                # BƯỚC 3.1: CHUẨN BỊ BATCH
                # ============================================================
                batch = batch.to(device)  # Move batch lên GPU
                # batch shape: [batch_size, 3] với 3 cột [head, tail, rel]
                
                # ============================================================
                # BƯỚC 3.2: NEGATIVE SAMPLING
                # ============================================================
                # Tạo negative samples cho mỗi positive triple
                # 
                # INPUT: positive triple (h, r, t)
                # OUTPUT: [positive, neg1, neg2, ..., negN]
                #   - neg samples: thay h hoặc t bằng entity khác
                #   - num_negative: số lượng negatives mỗi positive
                #   - strict: đảm bảo negatives không phải triples thật
                batch = tasks.negative_sampling(
                    train_graph,
                    batch,
                    cfg.task.num_negative,           # Ví dụ: 128
                    strict=cfg.task.strict_negative, # True = không corrupt thành triple thật
                )
                # batch sau khi sampling: [batch_size, 1 + num_negative, 3]
                
                # ============================================================
                # BƯỚC 3.3: FORWARD PASS - DỰ ĐOÁN SCORES
                # ============================================================
                # Model dự đoán score cho mỗi triple
                # Score cao = triple có khả năng đúng cao
                pred = parallel_model(train_graph, batch)
                # pred shape: [batch_size, 1 + num_negative]
                #   - Column 0: score của positive triple
                #   - Columns 1+: scores của negative triples
                
                # ============================================================
                # BƯỚC 3.4: TẠO TARGET LABELS
                # ============================================================
                # Target: one-hot vector với 1 ở vị trí positive, 0 ở negatives
                target = torch.zeros_like(pred)
                target[:, 0] = 1  # Chỉ positive (column 0) có label = 1
                # target shape: [batch_size, 1 + num_negative]
                
                # ============================================================
                # BƯỚC 3.5: TÍNH LOSS - BINARY CROSS ENTROPY
                # ============================================================
                # BCE loss: đo sự khác biệt giữa predictions và targets
                # reduction="none": tính loss riêng cho từng element
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                # loss shape: [batch_size, 1 + num_negative]
                
                # ============================================================
                # BƯỚC 3.6: ADVERSARIAL NEGATIVE SAMPLING (Tùy chọn)
                # ============================================================
                # Trọng số cao hơn cho "hard negatives" (negatives khó phân biệt)
                # Giúp model học tốt hơn
                neg_weight = torch.ones_like(pred)
                
                if cfg.task.adversarial_temperature > 0:
                    # Adversarial mode: Weight dựa trên scores của negatives
                    # Negatives có score cao (gần positive) được weight nhiều hơn
                    with torch.no_grad():  # Không tính gradient cho weights
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature, 
                            dim=-1
                        )
                    # Temperature cao → weights uniform hơn
                    # Temperature thấp → focus vào hard negatives
                else:
                    # Uniform weighting: tất cả negatives có weight bằng nhau
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                
                # ============================================================
                # BƯỚC 3.7: WEIGHTED LOSS
                # ============================================================
                # Nhân loss với weights và normalize
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                # Trung bình loss qua batch
                loss = loss.mean()
                
                # ============================================================
                # BƯỚC 3.8: BACKWARD PASS - TÍNH GRADIENTS
                # ============================================================
                loss.backward()  # Backpropagation
                
                # ============================================================
                # BƯỚC 3.9: UPDATE PARAMETERS
                # ============================================================
                optimizer.step()        # Cập nhật weights
                optimizer.zero_grad()   # Reset gradients về 0
                
                # ============================================================
                # BƯỚC 3.10: LOGGING
                # ============================================================
                if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.info(separator)
                    logger.info(f"binary cross entropy: {loss:g}")
                
                losses.append(loss.item())  # Lưu loss value
                batch_id += 1               # Tăng batch counter
        
        # ====================================================================
        # KẾT THÚC EPOCH - LOG AVERAGE LOSS
        # ====================================================================
        if utils.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.info(separator)
            logger.info(f"Epoch {epoch} end")
            logger.info(line)
            logger.info(f"average binary cross entropy: {avg_loss:g}")
        
        # ====================================================================
        # SYNCHRONIZE TẤT CẢ PROCESSES
        # ====================================================================
        # Đợi tất cả GPUs hoàn thành epoch trước khi validate
        utils.synchronize()
        
        # ====================================================================
        # VALIDATION
        # ====================================================================
        if rank == 0:
            logger.info(separator)
            logger.info("Evaluate on valid")
        
        # Gọi hàm test() để evaluate trên validation set
        result = test(
            cfg,
            model,
            dataset_loader,
            device=device,
        )
        # result: average MRR (Mean Reciprocal Rank) - metric chính
        
        # ====================================================================
        # SAVE BEST MODEL
        # ====================================================================
        if rank == 0:
            if result > best_result:
                # Cập nhật best result
                best_result = result
                best_epoch = epoch
                
                logger.info("Save checkpoint to model_best.pth")
                
                # Tạo state dictionary chứa model và optimizer
                state = {
                    "model": model.state_dict(),      # Model weights
                    "optimizer": optimizer.state_dict(),  # Optimizer state
                    "epoch": epoch,                   # Current epoch
                }
                
                # Save best model
                torch.save(
                    state, 
                    os.path.join(output_dir, "model_best.pth")
                )
        
        # ====================================================================
        # SAVE CHECKPOINT ĐỊNH KỲ
        # ====================================================================
        # Save checkpoint mỗi N epochs để có thể resume
        if rank == 0 and epoch % cfg.train.save_interval == 0:
            logger.info(f"Save checkpoint to model_epoch_{epoch}.pth")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                state, 
                os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            )
    
    # ========================================================================
    # KẾT THÚC TRAINING - LOG BEST RESULT
    # ========================================================================
    if rank == 0:
        logger.info(separator)
        logger.info(f"Best result: {best_result:g} at epoch {best_epoch}")


# ============================================================================
# HÀM 3: TEST - ĐÁNH GIÁ MODEL TRÊN TEST/VALIDATION SET
# ============================================================================
def test(
    cfg: DictConfig,
    model: nn.Module,
    dataset_loader: GraphDatasetLoader,
    device: torch.device,
    return_metrics: bool = False,
) -> float:
    """
    Đánh giá model trên test/validation datasets
    
    THAM SỐ:
        cfg: Configuration
        model: Model cần evaluate
        dataset_loader: Loader chứa test datasets
        device: CPU hoặc CUDA
        return_metrics: True = return dict metrics, False = return avg MRR
    
    RETURN:
        avg_mrr (float): Average Mean Reciprocal Rank qua tất cả datasets
        hoặc all_metrics (dict): Dictionary chứa tất cả metrics
    
    EVALUATION TASK:
        Link Prediction với filtered setting
        - Cho (h, r, ?): rank tail entity đúng trong tất cả entities
        - Cho (?, r, t): rank head entity đúng trong tất cả entities
    
    METRICS:
        - MR (Mean Rank): Trung bình rank của entity đúng
        - MRR (Mean Reciprocal Rank): Trung bình 1/rank
        - Hits@K: Tỷ lệ entity đúng nằm trong top-K
    """
    
    # ========================================================================
    # SETUP
    # ========================================================================
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    
    model.eval()  # Set model về evaluation mode (tắt dropout, etc.)
    
    all_mrr = []         # Lưu MRR của từng dataset
    all_metrics = {}     # Lưu tất cả metrics của từng dataset
    
    # ========================================================================
    # VÒNG LẶP QUA TỪNG TEST DATASET
    # ========================================================================
    for test_dataset in dataset_loader:
        # ====================================================================
        # CHUẨN BỊ TEST DATASET
        # ====================================================================
        test_dataset = create_kgc_dataset(
            test_dataset,
            cfg.train.batch_size,  # Dùng batch size từ config
            world_size,
            rank,
            is_train=False,        # Test mode
            shuffle=False,         # Không shuffle khi test
            fast_test=cfg.task.fast_test if "fast_test" in cfg.task else None,
        )
        
        test_data_name = test_dataset["data_name"]
        test_loader = test_dataset["data_loader"]
        test_graph = test_dataset["graph"].to(device)
        
        # filtered_data: Chứa tất cả triples đúng để filter evaluation
        filtered_data = test_dataset["val_filtered_data"]
        if filtered_data is not None:
            filtered_data = filtered_data.to(device)
        
        # ====================================================================
        # EVALUATION MODE - NO GRADIENT
        # ====================================================================
        model.eval()
        
        # Lists để collect rankings từ tất cả batches
        rankings = []          # Rankings cho cả head và tail prediction
        num_negatives = []     # Số negative samples cho mỗi triple
        tail_rankings = []     # Rankings chỉ cho tail prediction
        num_tail_negs = []     # Số negatives cho tail prediction
        
        # ====================================================================
        # VÒNG LẶP QUA BATCHES
        # ====================================================================
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            # batch: [batch_size, 3] - mỗi row là (h, t, r)
            
            # ================================================================
            # BƯỚC 1: TẠO ALL NEGATIVE SAMPLES
            # ================================================================
            # Thay vì sample ngẫu nhiên, test trên TẤT CẢ entities
            # 
            # t_batch: Test tail prediction - thay tail bằng tất cả entities
            # h_batch: Test head prediction - thay head bằng tất cả entities
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            
            # ================================================================
            # BƯỚC 2: FORWARD PASS - DỰ ĐOÁN SCORES
            # ================================================================
            with torch.no_grad():  # Không tính gradient khi test
                t_pred = model(test_graph, t_batch)  # Scores cho tail prediction
                h_pred = model(test_graph, h_batch)  # Scores cho head prediction
            # pred shape: [batch_size, num_entities]
            #   - Mỗi row: scores cho tất cả entities có thể
            
            # ================================================================
            # BƯỚC 3: TẠO MASK CHO FILTERED SETTING
            # ================================================================
            # Filtered setting: Loại bỏ các triples đúng khác khỏi ranking
            # Ví dụ: Nếu (Obama, president_of, USA) và (Obama, president_of, Kenya)
            #        đều đúng, khi test (Obama, president_of, ?), chỉ rank USA,
            #        không penalty Kenya
            
            if filtered_data is None:
                # Unfiltered: Chỉ loại bỏ positive triple đang test
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                # Filtered: Loại bỏ TẤT CẢ triples đúng trong toàn bộ KG
                t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
            # mask shape: [batch_size, num_entities]
            #   - True: entity hợp lệ để rank
            #   - False: entity là triple đúng, không rank
            
            # ================================================================
            # BƯỚC 4: EXTRACT POSITIVE INDICES
            # ================================================================
            # Lấy indices của head, tail, relation đúng từ batch
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            # pos_h_index: [batch_size] - head entities đúng
            # pos_t_index: [batch_size] - tail entities đúng
            # pos_r_index: [batch_size] - relation types
            
            # ================================================================
            # BƯỚC 5: TÍNH RANKINGS
            # ================================================================
            # Rank = vị trí của entity đúng khi sort scores giảm dần
            # Rank 1 = tốt nhất (entity đúng có score cao nhất)
            
            # Tail prediction ranking
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            # Head prediction ranking
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            
            # Đếm số negatives hợp lệ (sau khi filter)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)
            
            # ================================================================
            # BƯỚC 6: LƯU RANKINGS
            # ================================================================
            rankings += [t_ranking, h_ranking]  # Cả head và tail
            num_negatives += [num_t_negative, num_h_negative]
            
            tail_rankings += [t_ranking]  # Chỉ tail (cho một số metrics)
            num_tail_negs += [num_t_negative]
        
        # ====================================================================
        # AGGREGATE RANKINGS TỪ TẤT CẢ BATCHES
        # ====================================================================
        ranking = torch.cat(rankings)  # Concatenate tất cả rankings
        num_negative = torch.cat(num_negatives)
        
        # Để sync results giữa các GPUs trong distributed setting
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)  # Số rankings trên GPU này
        
        # Tương tự cho tail-only rankings
        tail_ranking = torch.cat(tail_rankings)
        num_tail_neg = torch.cat(num_tail_negs)
        all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size_t[rank] = len(tail_ranking)
        
        # ====================================================================
        # DISTRIBUTED: GATHER SIZES TỪ TẤT CẢ GPUS
        # ====================================================================
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)
        
        # ====================================================================
        # DISTRIBUTED: GATHER RANKINGS TỪ TẤT CẢ GPUS
        # ====================================================================
        # Cumulative sum để biết vị trí của mỗi GPU trong tensor tổng
        cum_size = all_size.cumsum(0)
        
        # Tạo tensor chứa tất cả rankings từ tất cả GPUs
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        # Đặt rankings của GPU này vào đúng vị trí
        all_ranking[cum_size[rank] - all_size[rank] : cum_size[rank]] = ranking
        
        # Tương tự cho num_negative
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank] : cum_size[rank]] = (
            num_negative
        )
        
        # Tương tự cho tail-only rankings
        cum_size_t = all_size_t.cumsum(0)
        all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
        all_ranking_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            tail_ranking
        )
        all_num_negative_t = torch.zeros(
            all_size_t.sum(), dtype=torch.long, device=device
        )
        all_num_negative_t[cum_size_t[rank] - all_size_t[rank] : cum_size_t[rank]] = (
            num_tail_neg
        )
        
        # ====================================================================
        # DISTRIBUTED: SYNC RANKINGS GIỮA CÁC GPUS
        # ====================================================================
        if world_size > 1:
            # all_reduce: Mỗi GPU có toàn bộ rankings từ tất cả GPUs
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)
        
        # ====================================================================
        # TÍNH CÁC METRICS (CHỈ TRÊN RANK 0)
        # ====================================================================
        metrics = {}
        if rank == 0:
            logger.info(f"{'-' * 15} Test on {test_data_name} {'-' * 15}")
            
            # Iterate qua các metrics được config
            for metric in cfg.task.metric:
                # ============================================================
                # XỬ LÝ TAIL-ONLY METRICS
                # ============================================================
                if "-tail" in metric:
                    # Metric chỉ tính trên tail prediction
                    _metric_name, direction = metric.split("-")
                    if direction != "tail":
                        raise ValueError(
                            "Only tail metric is supported in this mode"
                        )
                    _ranking = all_ranking_t
                    _num_neg = all_num_negative_t
                else:
                    # Metric tính trên cả head và tail
                    _ranking = all_ranking
                    _num_neg = all_num_negative
                    _metric_name = metric
                
                # ============================================================
                # TÍNH TỪNG METRIC
                # ============================================================
                if _metric_name == "mr":
                    # Mean Rank: Trung bình của rankings
                    # Càng thấp càng tốt
                    score = _ranking.float().mean()
                    
                elif _metric_name == "mrr":
                    # Mean Reciprocal Rank: Trung bình 1/rank
                    # Càng cao càng tốt (max = 1 khi rank = 1)
                    score = (1 / _ranking.float()).mean()
                    
                elif _metric_name.startswith("hits@"):
                    # Hits@K: Tỷ lệ entity đúng nằm trong top-K
                    # Ví dụ: hits@10 = % triples có rank <= 10
                    
                    # Parse K từ metric name
                    values = _metric_name[5:].split("_")
                    threshold = int(values[0])  # K
                    
                    if len(values) > 1:
                        # ====================================================
                        # UNBIASED HITS@K ESTIMATION
                        # ====================================================
                        # Khi dùng negative sampling (không test trên tất cả)
                        # cần ước lượng hits@K không bias
                        num_sample = int(values[1])
                        
                        # False positive rate: tỷ lệ negatives rank cao hơn positive
                        fp_rate = (_ranking - 1).float() / _num_neg
                        
                        score = 0
                        # Tính xác suất có <= threshold-1 false positives
                        for i in range(threshold):
                            # Binomial coefficient
                            num_comb = (
                                math.factorial(num_sample - 1)
                                / math.factorial(i)
                                / math.factorial(num_sample - i - 1)
                            )
                            # Binomial probability
                            score += (
                                num_comb
                                * (fp_rate**i)
                                * ((1 - fp_rate) ** (num_sample - i - 1))
                            )
                        score = score.mean()
                    else:
                        # ====================================================
                        # EXACT HITS@K
                        # ====================================================
                        # Đơn giản: đếm tỷ lệ rank <= K
                        score = (_ranking <= threshold).float().mean()
                
                # Log metric
                logger.info(f"{metric}: {score:g}")
                metrics[metric] = score
        
        # ====================================================================
        # TÍNH MRR CHO DATASET NÀY
        # ====================================================================
        mrr = (1 / all_ranking.float()).mean()
        all_mrr.append(mrr)
        all_metrics[test_data_name] = metrics
        
        if rank == 0:
            logger.info(separator)
    
    # ========================================================================
    # RETURN AVERAGE MRR HOẶC ALL METRICS
    # ========================================================================
    avg_mrr = sum(all_mrr) / len(all_mrr)
    return avg_mrr if not return_metrics else all_metrics


# ============================================================================
# HÀM MAIN - ĐIỂM VÀO CHÍNH
# ============================================================================
@hydra.main(config_path="config", config_name="stage2_kg_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point cho Stage 2: Pre-training trên Knowledge Graph
    
    LUỒNG CHÍNH:
    1. Setup distributed training environment
    2. Initialize datasets
    3. Create model
    4. Train and validate
    5. Final test
    6. Save pretrained model
    """
    
    # ========================================================================
    # BƯỚC 1: SETUP DISTRIBUTED TRAINING
    # ========================================================================
    # Khởi tạo distributed training (nếu dùng nhiều GPUs)
    # - Thiết lập process groups
    # - Sync giữa các processes
    utils.init_distributed_mode(cfg.train.timeout)
    
    # Set random seed (khác nhau cho mỗi rank để tăng diversity)
    torch.manual_seed(cfg.seed + utils.get_rank())
    
    # ========================================================================
    # BƯỚC 2: SETUP OUTPUT DIRECTORY
    # ========================================================================
    # Chỉ rank 0 tạo output directory
    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]
    
    # Broadcast output_dir từ rank 0 sang các ranks khác
    if utils.get_world_size() > 1:
        dist.broadcast_object_list(
            output_dir_list, src=0
        )
    output_dir = output_dir_list[0]
    
    # ========================================================================
    # BƯỚC 3: INITIALIZE DATASETS
    # ========================================================================
    if cfg.datasets.init_datasets:
        # Khởi tạo và process datasets (nếu chưa có)
        # Trả về relation embedding dimensions của mỗi dataset
        rel_emb_dim_list = utils.init_multi_dataset(
            cfg, utils.get_world_size(), utils.get_rank()
        )
        
        # Kiểm tra: Tất cả datasets phải có cùng relation embedding dim
        rel_emb_dim = set(rel_emb_dim_list)
        assert len(rel_emb_dim) == 1, (
            "All datasets should have the same relation embedding dimension"
        )
    else:
        # Datasets đã được initialize trước, dùng feat_dim từ config
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        rel_emb_dim = {cfg.datasets.feat_dim}
    
    # ========================================================================
    # BƯỚC 4: INITIALIZE MODEL
    # ========================================================================
    device = utils.get_device()  # Get CUDA device hoặc CPU
    
    # Instantiate model từ config với relation embedding dimension
    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())
    
    # ========================================================================
    # BƯỚC 5: LOAD CHECKPOINT (NẾU CÓ)
    # ========================================================================
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            # Load từ local file
            state = torch.load(
                cfg.train.checkpoint, 
                map_location="cpu", 
                weights_only=True
            )
            model.load_state_dict(state["model"])
        else:
            # Try load từ remote (Hugging Face, S3, etc.)
            model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)
    
    # Move model lên GPU
    model = model.to(device)
    
    # ========================================================================
    # LOG MODEL INFO
    # ========================================================================
    if utils.get_rank() == 0:
        # Đếm tổng số parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(line)
        logger.info(f"Number of parameters: {num_params}")
    
    # ========================================================================
    # BƯỚC 6: CREATE DATASET LOADER
    # ========================================================================
    # GraphDatasetLoader: Load nhiều datasets, có thể stream từ disk
    train_dataset_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,  # List các datasets để train
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    
    # ========================================================================
    # BƯỚC 7: TRAIN AND VALIDATE
    # ========================================================================
    train_and_validate(
        cfg,
        output_dir,
        model,
        dataset_loader=train_dataset_loader,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    
    # ========================================================================
    # BƯỚC 8: FINAL EVALUATION
    # ========================================================================
    if utils.get_rank() == 0:
        logger.info(separator)
        logger.info("Evaluate on valid")
    
    # Test trên validation set một lần nữa sau khi train xong
    test(cfg, model, train_dataset_loader, device=device)
    
    # ========================================================================
    # BƯỚC 9: SAVE PRETRAINED MODEL
    # ========================================================================
    # Save model theo format chuẩn để dùng cho inference sau
    if utils.is_main_process() and cfg.train.save_pretrained:
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    # Shutdown dataset loaders
    train_dataset_loader.shutdown()
    
    # Sync tất cả processes
    utils.synchronize()
    
    # Cleanup distributed training resources
    utils.cleanup()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()


"""
===============================================================================
TÓM TẮT LUỒNG CHẠY STAGE 2
===============================================================================

START
  │
  ├─► Setup Distributed Training
  │   ├─► Init process groups
  │   └─► Set random seeds
  │
  ├─► Setup Output Directory
  │   └─► Broadcast từ rank 0 sang các ranks khác
  │
  ├─► Initialize Datasets
  │   ├─► Load KG datasets từ stage 1
  │   └─► Verify relation embedding dimensions
  │
  ├─► Create Model
  │   ├─► Instantiate từ config
  │   └─► Load checkpoint (nếu có)
  │
  ├─► Training Loop (nhiều epochs)
  │   │
  │   ├─► For each dataset:
  │   │   │
  │   │   ├─► For each batch:
  │   │   │   ├─► Positive triple: (h, r, t)
  │   │   │   ├─► Negative sampling: tạo (h, r, t') hoặc (h', r, t)
  │   │   │   ├─► Forward: predict scores
  │   │   │   ├─► Loss: BCE với adversarial weighting
  │   │   │   ├─► Backward: compute gradients
  │   │   │   └─► Update: optimizer step
  │   │   │
  │   │   └─► Log training loss
  │   │
  │   ├─► Validation
  │   │   ├─► Test trên validation set
  │   │   ├─► Compute metrics (MRR, Hits@K)
  │   │   └─► Save best model
  │   │
  │   └─► Save checkpoint định kỳ
  │
  ├─► Final Evaluation
  │   ├─► Test trên test set
  │   └─► Log final metrics
  │
  ├─► Save Pretrained Model
  │   └─► Format chuẩn cho inference
  │
  └─► Cleanup
      ├─► Shutdown loaders
      └─► Cleanup distributed resources
  
END

===============================================================================
NHIỆM VỤ CHÍNH: LINK PREDICTION / KNOWLEDGE GRAPH COMPLETION
===============================================================================

TASK: Cho một triple không đầy đủ, dự đoán entity còn thiếu

VÍ DỤ:
Input:  (Barack_Obama, president_of, ?)
Output: USA (rank 1), Kenya (rank 543), Vietnam (rank 1205), ...

TRAINING:
- Positive: (Obama, president_of, USA) → score cao
- Negatives: (Obama, president_of, Vietnam) → score thấp
- Model học phân biệt triples đúng và sai

EVALUATION:
- Rank entity đúng trong tất cả entities
- Metrics: MRR, MR, Hits@1, Hits@3, Hits@10

===============================================================================
CÁC COMPONENT CHÍNH
===============================================================================

1. Negative Sampling:
   - Tạo negative triples bằng cách corrupt head hoặc tail
   - Adversarial sampling: focus vào hard negatives
   - Strict negative: đảm bảo không corrupt thành triple thật

2. Model Architecture:
   - Input: graph structure + triple
   - Output: score (xác suất triple đúng)
   - Thường dùng: GNN-based, TransE-like, etc.

3. Training Loss:
   - Binary Cross Entropy
   - Positive sample có target = 1
   - Negative samples có target = 0

4. Evaluation:
   - Filtered setting: loại bỏ triples đúng khác khỏi ranking
   - Unfiltered: rank trên tất cả entities
   - Metrics đo lường chất lượng ranking

5. Distributed Training:
   - Multi-GPU với DistributedDataParallel
   - Mỗi GPU train trên một phần data
   - Sync gradients và results

===============================================================================
MỤC ĐÍCH CỦA STAGE 2
===============================================================================

Pre-train model trên Knowledge Graph:
✓ Học embeddings tốt cho entities và relations
✓ Model hiểu cấu trúc và semantics của KG
✓ Pretrained model làm backbone cho downstream tasks

Sau stage 2:
✓ Model đã học biểu diễn của entities/relations
✓ Có thể fine-tune cho QA, retrieval, reasoning tasks
✓ Saved pretrained weights để tái sử dụng

===============================================================================
"""