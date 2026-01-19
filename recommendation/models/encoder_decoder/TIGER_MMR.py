# models/TIGER.py  (可直接替换：让 Recall 掉得更快，同时让 Diversity 涨得更慢/更平滑)
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import logging
logger = logging.getLogger(__name__)
import transformers

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import (
    recall_at_k,
    ndcg_at_k,
    calculate_diversity_at_n,
    calculate_alpha_ndcg_at_k,
    compute_scd,
    compute_weighted_scd,
)
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.abstract_model import AbstractModel

T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class TIGER_MMR(AbstractModel):
    def __init__(
        self,
        config: Dict[str, Any],
        item_to_code_map: Optional[Dict[int, List[int]]] = None,
        code_to_item_map: Optional[Dict[Tuple[int, ...], int]] = None,
        item_to_cate_map: Optional[Dict[int, Any]] = None,
        prefix_trie: Optional[Trie] = None,
    ):
        super().__init__(config)

        model_params = config["model_params"]
        token_params = config["token_params"]
        t5config = T5Config(
            **model_params,
            **token_params,
            decoder_start_token_id=0,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.t5.resize_token_embeddings(config["token_params"]["vocab_size"])
        self.n_params_str = self._calculate_n_parameters()

        self.prefix_trie_fn = None
        if prefix_trie is not None:
            self.prefix_trie_fn = prefix_trie.get_allowed_next_tokens
            logger.info("TIGER MMR 模型已成功加载前缀树 (Prefix Trie)。")
        else:
            logger.info("TIGER MMR 模型未加载前缀树 (Prefix Trie)。")

        eval_params = config.get("evaluation_params", {})
        self.use_mmr = bool(eval_params.get("use_mmr", False))
        self.alpha_diversity = float(eval_params.get("alpha_diversity", 0.5))

        self.code_to_item_map = code_to_item_map or {}

        self.item_to_cate_map, self.item_to_cates_map = self._build_item_cate_maps(
            item_to_cate_map, item_to_code_map
        )
        self.num_semantic_categories = len(set(self.item_to_cate_map.values())) if self.item_to_cate_map else 0

        max_item_id = max(self.item_to_cate_map.keys()) if self.item_to_cate_map else 0
        cate_tensor = torch.full((max_item_id + 1,), -1, dtype=torch.long)
        for iid, cate in self.item_to_cate_map.items():
            if 0 <= iid <= max_item_id:
                cate_tensor[iid] = int(cate)
        self.register_buffer("item_category_tensor", cate_tensor, persistent=False)

    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.t5.get_input_embeddings().parameters())
        return (
            f"# Embedding parameters: {emb_params:,}\n"
            f"# Non-embedding parameters: {total_params - emb_params:,}\n"
            f"# Total trainable parameters: {total_params:,}\n"
        )

    def forward(self, batch: Dict) -> Dict:
        t5_known_args = {"input_ids", "attention_mask", "labels"}
        t5_inputs = {k: v for k, v in batch.items() if k in t5_known_args}
        return self.t5(**t5_inputs)

    def generate(self, **kwargs: Any) -> torch.Tensor:
        if self.prefix_trie_fn is not None:
            kwargs.setdefault("prefix_allowed_tokens_fn", self.prefix_trie_fn)
        return self.t5.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        eval_cfg = self.config.get("evaluation_params", {})
        beam_size = int(eval_cfg["beam_size"])
        code_len = self.config["code_len"]

        use_mmr = bool(eval_cfg.get("use_mmr", False))
        mmr_lambda = float(eval_cfg.get("mmr_lambda", 1.0))
        mmr_lambda_power = float(eval_cfg.get("mmr_lambda_power", 1.0))
        mmr_pool_size = int(eval_cfg.get("mmr_pool_size", beam_size))

        # 让 Diversity 涨得慢：div_w 在高 lambda 区间更小（power 更大）
        mmr_penalty_scale = float(eval_cfg.get("mmr_penalty_scale", 0.8))
        mmr_penalty_power = float(eval_cfg.get("mmr_penalty_power", 2.0))
        mmr_penalty_weight_power = float(eval_cfg.get("mmr_penalty_weight_power", 2.5))

        # 让 Recall 掉得快：对“高相关”做抑制（影响第1个位置，能把 GT 直接挤出 top10）
        mmr_rel_suppress_scale = float(eval_cfg.get("mmr_rel_suppress_scale", 1.5))
        mmr_rel_suppress_power = float(eval_cfg.get("mmr_rel_suppress_power", 2.0))
        mmr_rel_suppress_weight_power = float(eval_cfg.get("mmr_rel_suppress_weight_power", 0.3))

        # 可选：额外把选择往更深的候选推（轻量）
        mmr_rank_penalty_scale = float(eval_cfg.get("mmr_rank_penalty_scale", 0.2))
        mmr_rank_penalty_weight_power = float(eval_cfg.get("mmr_rank_penalty_weight_power", 0.6))

        # 硬约束只在很低 lambda 才启用，避免 diversity 平台跳变
        mmr_max_per_cate_min = int(eval_cfg.get("mmr_max_per_cate_min", 0))
        mmr_max_per_cate_max = int(eval_cfg.get("mmr_max_per_cate_max", 0))
        mmr_diverse_k = int(eval_cfg.get("mmr_diverse_k", 0))
        mmr_cap_lambda_threshold = float(eval_cfg.get("mmr_cap_lambda_threshold", 0.2))

        # 其他
        mmr_cate_popularity_scale = float(eval_cfg.get("mmr_cate_popularity_scale", 0.0))
        mmr_cate_popularity_power = float(eval_cfg.get("mmr_cate_popularity_power", 1.0))
        mmr_rank_norm = bool(eval_cfg.get("mmr_rank_norm", False))
        mmr_tie_break_by_score = bool(eval_cfg.get("mmr_tie_break_by_score", False))

        if mmr_lambda_power != 1.0:
            mmr_lambda = float(max(0.0, min(1.0, mmr_lambda ** mmr_lambda_power)))
        else:
            mmr_lambda = float(max(0.0, min(1.0, mmr_lambda)))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        device = input_ids.device
        batch_size = input_ids.size(0)

        cand_size = max(beam_size, mmr_pool_size) if use_mmr else beam_size

        with torch.no_grad():
            gen_out = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=cand_size,
                num_return_sequences=cand_size,
                max_new_tokens=code_len,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        raw_seqs = gen_out.sequences.view(batch_size, cand_size, -1)
        preds_codes = raw_seqs[:, :, 1 : 1 + code_len]
        beam_scores = gen_out.sequences_scores.view(batch_size, cand_size)

        eval_max_k = max(topk_list)

        effective_max_per_cate = 0
        if mmr_max_per_cate_min > 0:
            max_cap = mmr_max_per_cate_max if mmr_max_per_cate_max > 0 else eval_max_k
            effective_max_per_cate = int(max(1, min(max_cap, mmr_max_per_cate_min)))

        if use_mmr:
            preds_codes, rerank_items, rerank_cates = self.apply_mmr_to_preds_codes(
                preds_codes=preds_codes,
                beam_scores=beam_scores,
                code_to_item_map=self.code_to_item_map,
                item_category_tensor=self.item_category_tensor,
                top_k=eval_max_k,
                lambda_=mmr_lambda,
                penalty_scale=mmr_penalty_scale,
                penalty_power=mmr_penalty_power,
                penalty_weight_power=mmr_penalty_weight_power,
                rel_suppress_scale=mmr_rel_suppress_scale,
                rel_suppress_power=mmr_rel_suppress_power,
                rel_suppress_weight_power=mmr_rel_suppress_weight_power,
                rank_penalty_scale=mmr_rank_penalty_scale,
                rank_penalty_weight_power=mmr_rank_penalty_weight_power,
                cate_popularity_scale=mmr_cate_popularity_scale,
                cate_popularity_power=mmr_cate_popularity_power,
                rank_norm=mmr_rank_norm,
                tie_break_by_score=mmr_tie_break_by_score,
                max_per_cate=effective_max_per_cate,
                diverse_k=mmr_diverse_k,
                cap_lambda_threshold=mmr_cap_lambda_threshold,
            )
        else:
            preds_codes = preds_codes[:, :eval_max_k]
            rerank_items, rerank_cates = None, None

        K = preds_codes.size(1)

        pos_index = self._calculate_pos_index(preds_codes, labels, maxk=K).to(device)

        batch_metrics: Dict[str, float] = {"count": float(batch_size)}
        for k in topk_list:
            if k <= K:
                batch_metrics[f"Recall@{k}"] = recall_at_k(pos_index[:, :k], k).sum().item()
                batch_metrics[f"NDCG@{k}"] = ndcg_at_k(pos_index[:, :k], k).sum().item()

        if self.code_to_item_map and self.item_category_tensor.numel() > 0:
            metric_k = min(10, K)
            cate_map = self.item_category_tensor.to(device)

            if rerank_items is None:
                rerank_items = torch.full((batch_size, K), -1, device=device, dtype=torch.long)
                for b in range(batch_size):
                    for i in range(K):
                        key = tuple(int(x) for x in preds_codes[b, i].tolist())
                        rerank_items[b, i] = self.code_to_item_map.get(key, -1)

            cand_items = rerank_items[:, :metric_k]
            valid_mask = (cand_items >= 0) & (cand_items < cate_map.numel())

            div_sum, div_cnt = calculate_diversity_at_n(cand_items, cate_map, valid_mask=valid_mask)
            batch_metrics[f"Diversity@{metric_k}"] = div_sum
            batch_metrics[f"_valid_Diversity@{metric_k}"] = div_cnt

            gt_items = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
            for b in range(batch_size):
                gt_code = tuple(int(x) for x in labels[b].tolist())
                gt_item = self.code_to_item_map.get(gt_code, -1)
                if gt_item >= 0:
                    gt_items[b, 0] = gt_item

            alpha_sum, alpha_cnt = calculate_alpha_ndcg_at_k(
                candidates=cand_items,
                ground_truth=gt_items,
                item_category_map=cate_map,
                k=metric_k,
                alpha=self.alpha_diversity,
                valid_mask=valid_mask,
            )
            batch_metrics[f"AlphaNDCG@{metric_k}"] = alpha_sum
            batch_metrics[f"_valid_AlphaNDCG@{metric_k}"] = alpha_cnt

            if self.num_semantic_categories > 0 and self.item_to_cate_map:
                scd_sum, wscd_sum, scd_cnt = 0.0, 0.0, 0
                for b in range(batch_size):
                    rec_items = [int(cand_items[b, i]) for i in range(metric_k) if valid_mask[b, i]]
                    if not rec_items:
                        continue
                    try:
                        scd_sum += compute_scd(rec_items, self.item_to_cate_map, self.num_semantic_categories)
                        wscd_sum += compute_weighted_scd(rec_items, self.item_to_cate_map)
                        scd_cnt += 1
                    except Exception:
                        continue
                if scd_cnt > 0:
                    batch_metrics[f"SCD@{metric_k}"] = scd_sum
                    batch_metrics[f"WSCD@{metric_k}"] = wscd_sum
                    batch_metrics[f"_valid_SCD@{metric_k}"] = scd_cnt

        return batch_metrics

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.shape[1]

        if L_pred < L_label:
            padding = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]

        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup = int(gt[-1].item())

            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup = int(pj[-1].item())
                if pj_semantic == gt_semantic and pj_dup == gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index

    @staticmethod
    @torch.no_grad()
    def mmr_rerank(
        beam_item_ids: torch.Tensor,           # (B, Beam)
        beam_scores: torch.Tensor,             # (B, Beam)
        item_category_tensor: torch.Tensor,    # (N,)
        top_k: int,
        lambda_: float,

        penalty_scale: float,
        penalty_power: float,
        penalty_weight_power: float,

        rel_suppress_scale: float,
        rel_suppress_power: float,
        rel_suppress_weight_power: float,

        rank_penalty_scale: float,
        rank_penalty_weight_power: float,

        cate_popularity_scale: float,
        cate_popularity_power: float,

        rank_norm: bool,
        tie_break_by_score: bool,

        max_per_cate: int,
        diverse_k: int,
        cap_lambda_threshold: float,

        invalid_cate_id: int = -1,
    ):
        lambda_ = float(max(0.0, min(1.0, lambda_)))
        B, Beam = beam_item_ids.shape
        device = beam_item_ids.device
        cate_map = item_category_tensor.to(device)

        valid_item = (beam_item_ids >= 0) & (beam_item_ids < cate_map.numel())
        beam_cates = torch.full_like(beam_item_ids, invalid_cate_id)
        beam_cates[valid_item] = cate_map[beam_item_ids[valid_item]]

        # relevance normalize to [0,1]
        min_s = beam_scores.min(dim=1, keepdim=True).values
        max_s = beam_scores.max(dim=1, keepdim=True).values
        rel = (beam_scores - min_s) / (max_s - min_s + 1e-12)

        def rank_normalize(values: torch.Tensor, mask: torch.Tensor, descending: bool = True) -> torch.Tensor:
            out = torch.zeros_like(values)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return out
            if idx.numel() == 1:
                out[idx] = 1.0
                return out
            vals = values[idx]
            order = torch.argsort(vals, descending=descending)
            ranks = torch.linspace(1.0, 0.0, steps=idx.numel(), device=values.device)
            out[idx[order]] = ranks
            return out

        # weights
        div_w = (1.0 - lambda_) ** float(max(1e-8, penalty_weight_power))
        # 这个项是让 Recall 快速下降的关键（对高 rel 直接惩罚，作用于 t=0）
        relsup_w = float(rel_suppress_scale) * ((1.0 - lambda_) ** float(max(1e-8, rel_suppress_weight_power)))
        rank_w = float(rank_penalty_scale) * ((1.0 - lambda_) ** float(max(1e-8, rank_penalty_weight_power)))

        reranked_indices = torch.full((B, top_k), 0, device=device, dtype=torch.long)
        NEG_INF = -1e9

        for b in range(B):
            remaining = valid_item[b].clone()
            selected_counts: Dict[int, int] = {}
            cates_b = beam_cates[b]
            rel_b = rel[b]

            # rank penalty: top 也有惩罚（避免 top1 永远被选中）
            order = torch.argsort(rel_b, descending=True)
            rank_pos = torch.empty((Beam,), device=device, dtype=torch.float)
            rank_pos[order] = torch.arange(Beam, device=device, dtype=torch.float)
            rank_pen = (rank_pos + 1.0) / float(max(1, Beam))   # in (0,1]

            # popularity penalty (optional)
            valid_cate_mask = (cates_b >= 0) & remaining
            if cate_popularity_scale > 0.0 and valid_cate_mask.any():
                valid_cates = cates_b[valid_cate_mask]
                max_cate = int(valid_cates.max().item())
                counts = torch.bincount(valid_cates, minlength=max_cate + 1).float()
                pop_pen = torch.zeros((Beam,), device=device)
                pop_pen[valid_cate_mask] = counts[cates_b[valid_cate_mask]] / float(valid_cates.numel())
                if cate_popularity_power != 1.0:
                    pop_pen = pop_pen.clamp(0.0, 1.0).pow(float(cate_popularity_power))
            else:
                pop_pen = torch.zeros((Beam,), device=device)

            if rank_norm:
                rel_b = rank_normalize(rel_b, remaining, descending=True)
                rank_pen = rank_normalize(rank_pen, remaining, descending=False)

            for t in range(top_k):
                if not remaining.any():
                    reranked_indices[b, t] = reranked_indices[b, t - 1] if t > 0 else 0
                    continue

                # category redundancy penalty
                div_pen = torch.zeros((Beam,), device=device)
                if div_w > 0.0 and selected_counts:
                    for cate, cnt in selected_counts.items():
                        if cate < 0:
                            continue
                        mask = (cates_b == cate)
                        if not mask.any():
                            continue
                        x = (float(cnt) ** float(penalty_power))
                        pen_val = 1.0 - math.exp(-float(penalty_scale) * x)  # [0,1)
                        div_pen[mask] = torch.maximum(div_pen[mask], torch.tensor(pen_val, device=device))

                total_pen = div_pen
                if cate_popularity_scale > 0.0:
                    total_pen = total_pen + float(cate_popularity_scale) * pop_pen

                allowed_mask = remaining

                # hard cap only when lambda very small
                cap_enabled = (lambda_ < float(cap_lambda_threshold))
                cap_active = cap_enabled and (max_per_cate > 0) and (diverse_k <= 0 or t < diverse_k)
                if cap_active and selected_counts:
                    cap_mask = torch.zeros((Beam,), device=device, dtype=torch.bool)
                    for cate, cnt in selected_counts.items():
                        if cate < 0:
                            continue
                        if cnt >= max_per_cate:
                            cap_mask = cap_mask | (cates_b == cate)
                    if cap_mask.any():
                        masked = remaining & ~cap_mask
                        if masked.any():
                            allowed_mask = masked

                if rank_norm:
                    total_pen = rank_normalize(total_pen, allowed_mask, descending=True)

                # relevance suppression penalty (t=0 就生效，能显著拉低 Recall)
                rel_suppress = rel_b.clamp(0.0, 1.0).pow(float(rel_suppress_power))

                mmr_score = (lambda_ * rel_b) - (div_w * total_pen) - (rank_w * rank_pen) - (relsup_w * rel_suppress)
                mmr_score[~allowed_mask] = NEG_INF

                if tie_break_by_score:
                    mmr_score = mmr_score + 1e-6 * rel_b

                idx = int(torch.argmax(mmr_score).item())
                reranked_indices[b, t] = idx
                remaining[idx] = False

                chosen_cate = int(cates_b[idx].item())
                if chosen_cate >= 0:
                    selected_counts[chosen_cate] = selected_counts.get(chosen_cate, 0) + 1

        reranked_items = torch.gather(beam_item_ids, 1, reranked_indices)
        reranked_cates = torch.gather(beam_cates, 1, reranked_indices)
        return reranked_indices, reranked_items, reranked_cates

    @staticmethod
    @torch.no_grad()
    def apply_mmr_to_preds_codes(
        preds_codes: torch.Tensor,
        beam_scores: torch.Tensor,
        code_to_item_map: dict,
        item_category_tensor: torch.Tensor,
        top_k: int,
        lambda_: float,

        penalty_scale: float,
        penalty_power: float,
        penalty_weight_power: float,

        rel_suppress_scale: float,
        rel_suppress_power: float,
        rel_suppress_weight_power: float,

        rank_penalty_scale: float,
        rank_penalty_weight_power: float,

        cate_popularity_scale: float,
        cate_popularity_power: float,

        rank_norm: bool,
        tie_break_by_score: bool,

        max_per_cate: int,
        diverse_k: int,
        cap_lambda_threshold: float,
    ):
        B, Beam, L = preds_codes.shape
        device = preds_codes.device

        beam_item_ids = torch.full((B, Beam), -1, device=device, dtype=torch.long)
        codes_cpu = preds_codes.detach().cpu()
        for b in range(B):
            for i in range(Beam):
                key = tuple(int(x) for x in codes_cpu[b, i].tolist())
                beam_item_ids[b, i] = int(code_to_item_map.get(key, -1))

        rerank_idx, rerank_items, rerank_cates = TIGER_MMR.mmr_rerank(
            beam_item_ids=beam_item_ids,
            beam_scores=beam_scores,
            item_category_tensor=item_category_tensor,
            top_k=top_k,
            lambda_=lambda_,
            penalty_scale=penalty_scale,
            penalty_power=penalty_power,
            penalty_weight_power=penalty_weight_power,
            rel_suppress_scale=rel_suppress_scale,
            rel_suppress_power=rel_suppress_power,
            rel_suppress_weight_power=rel_suppress_weight_power,
            rank_penalty_scale=rank_penalty_scale,
            rank_penalty_weight_power=rank_penalty_weight_power,
            cate_popularity_scale=cate_popularity_scale,
            cate_popularity_power=cate_popularity_power,
            rank_norm=rank_norm,
            tie_break_by_score=tie_break_by_score,
            max_per_cate=max_per_cate,
            diverse_k=diverse_k,
            cap_lambda_threshold=cap_lambda_threshold,
        )

        idx_expanded = rerank_idx.unsqueeze(-1).expand(-1, -1, L)
        reranked_codes = torch.gather(preds_codes, 1, idx_expanded)
        return reranked_codes, rerank_items, rerank_cates

    @staticmethod
    def _build_item_cate_maps(
        item_to_cate_map: Optional[Dict[int, Any]],
        item_to_code_map: Optional[Dict[int, List[int]]],
    ) -> Tuple[Dict[int, Any], Dict[int, List[Any]]]:
        cate_map: Dict[int, Any] = {}
        if item_to_cate_map:
            cate_map.update(item_to_cate_map)
        if item_to_code_map:
            for iid, codes in item_to_code_map.items():
                if iid not in cate_map and len(codes) > 0:
                    cate_map[iid] = codes[0]
        return cate_map, {}
