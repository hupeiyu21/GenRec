# metrics.py

import math
from typing import List, Tuple

import torch

def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    根据命中索引计算 Recall@k。
    
    Args:
        pos_index (torch.Tensor): 形状为 (B, maxk) 的布尔张量，True 表示命中。
        k (int): top-k 的 k 值。

    Returns:
        torch.Tensor: 形状为 (B,) 的每个样本的 Recall@k 值。
    """
    return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    根据命中索引计算 NDCG@k。
    假设每个样本只有一个正例。
    
    Args:
        pos_index (torch.Tensor): 形状为 (B, maxk) 的布尔张量，True 表示命中。
        k (int): top-k 的 k 值。

    Returns:
        torch.Tensor: 形状为 (B,) 的每个样本的 NDCG@k 值。
    """
    # 排名从 1 开始
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device, dtype=torch.float)
    
    # 计算 DCG，只有命中位置有值
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, device=dcg.device))
    
    # IDCG@k 在此场景下恒为 1.0（因为只有一个正例，且理想排名是1）
    # 所以 NDCG@k == DCG@k
    return dcg[:, :k].sum(dim=1).cpu().float()


# --- Semantic Coverage Diversity (SCD) --- # added for SCD
Semantic = torch.Tensor


def compute_scd(
    recommended_items: List[int],
    item_to_semantic: dict,
    num_semantic_categories: int,
) -> float:
    """
    Semantic Coverage Diversity (SCD)
    SCD(R) = (#unique_semantic_categories_in_R) / (total_semantic_categories)
    """
    if num_semantic_categories <= 0:
        raise ValueError("num_semantic_categories must be positive")
    if recommended_items is None or len(recommended_items) == 0:
        raise ValueError("recommended_items is empty")
    unique_semantics = set()
    for item in recommended_items:
        if item not in item_to_semantic:
            raise KeyError(f"Item {item} not found in item_to_semantic.")
        semantic = item_to_semantic[item]
        if semantic is None:
            raise ValueError(f"Item {item} has no semantic category.")
        unique_semantics.add(semantic)
    return len(unique_semantics) / float(num_semantic_categories)


def compute_weighted_scd(
    recommended_items: List[int],
    item_to_semantic: dict,
) -> float:
    """
    Weighted SCD (WSCD)
    WSCD(R) = Σ_i ( 1 / log2(i + 2) ) * 1(category_i is first occurrence)
    """
    if recommended_items is None or len(recommended_items) == 0:
        raise ValueError("recommended_items is empty")
    seen = set()
    score = 0.0
    for idx, item in enumerate(recommended_items):
        if item not in item_to_semantic:
            raise KeyError(f"Item {item} not found in item_to_semantic.")
        semantic = item_to_semantic[item]
        if semantic is None:
            raise ValueError(f"Item {item} has no semantic category.")
        if semantic in seen:
            continue
        seen.add(semantic)
        score += 1.0 / math.log2(idx + 2)
    return score


def calculate_diversity_at_n(
    candidates: torch.Tensor,
    item_category_map: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> Tuple[float, int]:
    """
    Diversity@N (Intra-List Diversity)
    计算推荐列表中两两物品类别不相同的比例，返回「分數總和」和「有效樣本數」。

    Args:
        candidates: [B, N] long tensor, 预测的 Item IDs
        item_category_map: [Vocab_Size] long tensor, Item ID 到 Category ID 的映射
        valid_mask: [B, N] bool tensor，True 表示該位置的 item 有效

    Returns:
        (score_sum, valid_users): score_sum 為逐用戶得分之和，valid_users 為有效樣本數
    """
    if candidates.numel() == 0:
        return 0.0, 0

    if item_category_map.device != candidates.device:
        item_category_map = item_category_map.to(candidates.device)

    if valid_mask is None:
        valid_mask = candidates >= 0

    total_score = 0.0
    valid_users = 0

    B, N = candidates.shape
    for b in range(B):
        mask = valid_mask[b]
        if mask.sum() < 2:
            continue
        row_items = candidates[b][mask]
        # 防止越界
        safe_items = row_items.clone()
        safe_items = torch.clamp(safe_items, min=0, max=item_category_map.numel() - 1)
        cats = item_category_map[safe_items]
        cats = cats[cats >= 0]
        if cats.numel() < 2:
            continue
        diff = (cats.unsqueeze(1) != cats.unsqueeze(0)).sum().item()
        denom = float(cats.numel() * (cats.numel() - 1))
        total_score += diff / denom
        valid_users += 1

    return float(total_score), valid_users

import math
import torch
from typing import Tuple


def calculate_alpha_ndcg_at_k(
    candidates: torch.Tensor,          # (B, K)
    ground_truth: torch.Tensor,         # (B, 1) or (B, G)
    item_category_map: torch.Tensor,    # (num_items,)
    k: int,
    alpha: float = 0.5,
    valid_mask: torch.Tensor = None,
) -> Tuple[float, int]:
    """
    Alpha-NDCG@K (STRICT, non-inflated version)

    Key properties:
    - Relevance is ITEM-level (must hit GT item)
    - Alpha redundancy penalty is CATEGORY-level
    - Safe for single-GT-item setting
    - Alpha-NDCG will be comparable to standard NDCG
    """

    if candidates.numel() == 0:
        return 0.0, 0

    B, K_pred = candidates.shape
    eval_k = min(k, K_pred)

    cand_list = candidates.detach().cpu().tolist()
    gt_list = ground_truth.detach().cpu().tolist()
    cate_map = item_category_map.detach().cpu()

    if valid_mask is None:
        valid_mask = torch.ones_like(candidates, dtype=torch.bool)
    valid_list = valid_mask.detach().cpu().tolist()

    scores = []

    for b in range(B):
        # -----------------------------
        # 1. GT items (STRICT)
        # -----------------------------
        gt_items = {int(x) for x in gt_list[b] if x >= 0}
        if not gt_items:
            continue

        # GT categories (for redundancy penalty only)
        gt_cates = {
            int(cate_map[i].item())
            for i in gt_items
            if 0 <= i < len(cate_map) and int(cate_map[i].item()) >= 0
        }

        if not gt_cates:
            continue

        # -----------------------------
        # 2. Candidate list
        # -----------------------------
        cand_row = []
        for idx, (itm, is_valid) in enumerate(zip(cand_list[b], valid_list[b])):
            if idx >= eval_k:
                break
            if not is_valid or itm < 0 or itm >= len(cate_map):
                continue
            cand_row.append(int(itm))

        if not cand_row:
            continue

        # -----------------------------
        # 3. DCG (item-level relevance)
        # -----------------------------
        dcg = 0.0
        cate_counts = {}

        for rank, item_id in enumerate(cand_row, start=1):
            # ❗ STRICT relevance: must hit GT item
            if item_id not in gt_items:
                continue

            cate = int(cate_map[item_id].item())
            if cate < 0:
                continue

            # Alpha redundancy penalty by category
            gain = (1.0 - alpha) ** cate_counts.get(cate, 0)
            dcg += gain / math.log2(rank + 1)
            cate_counts[cate] = cate_counts.get(cate, 0) + 1

        # -----------------------------
        # 4. IDCG (ideal ranking)
        # -----------------------------
        # Ideal: GT items ranked at top, category redundancy penalized
        idcg = 0.0
        ideal_counts = {}

        ideal_items = list(gt_items)
        ideal_len = min(eval_k, len(ideal_items))

        for rank in range(1, ideal_len + 1):
            item_id = ideal_items[rank - 1]
            cate = int(cate_map[item_id].item())

            gain = (1.0 - alpha) ** ideal_counts.get(cate, 0)
            idcg += gain / math.log2(rank + 1)
            ideal_counts[cate] = ideal_counts.get(cate, 0) + 1

        if idcg > 0:
            scores.append(dcg / idcg)

    return float(sum(scores)), len(scores)
