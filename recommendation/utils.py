# utils.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import random
import torch
import logging
from pathlib import Path
import importlib
from collections.abc import Mapping
from collections import Counter
import json

VALID_QUANT_METHODS = {"rkmeans", "rvq", "rqvae", "opq", "pq", 'vqvae', 'mm_rqvae'}

def _ensure_dir_exists(dir_path: Path):
    """确保目录存在"""
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

def _recursive_update(base_dict: dict, new_dict: dict) -> dict:
    """
    遞迴地更新字典。
    如果 new_dict 中的鍵在 base_dict 中也存在且對應的值都是字典，
    則遞迴地合併它們，否則直接用 new_dict 的值覆蓋 base_dict 的值。
    """
    for key, value in new_dict.items():
        if isinstance(value, Mapping) and key in base_dict and isinstance(base_dict[key], Mapping):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _load_quant_details(path: str, quant_method: str) -> dict:
    """
    從指定的 YAML 檔案中，根據 quant_method 載入對應的參數。
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"[Config] 根據約定，未找到量化設定檔: {path}")
    
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if quant_method not in cfg or 'model_params' not in cfg[quant_method]:
        raise ValueError(f"[Config] 在 {path} 中缺少 '{quant_method}.model_params' 節點")
    
    mp = cfg[quant_method]['model_params']
    required_keys = ['codebook_size', 'num_levels']
    if not all(key in mp for key in required_keys):
         raise ValueError(f"[Config] 在 {path} 的 model_params 中缺少 'codebook_size' 或 'num_levels'")

    return mp


def load_and_process_config(model_name: str, dataset_name: str, quant_method: str, embedding_modality: str = 'text') -> dict:
    """
    通用配置加载器 (V6 - 支援 base.yaml 繼承與覆蓋)。
    """
    # === 1. ✅ 關鍵改動 2：依序載入 base 和 model-specific 設定檔 ===
    # 載入基礎設定檔
    base_config_path = Path("configs/base.yaml")
    if not base_config_path.is_file():
        raise FileNotFoundError(f"基礎設定檔未找到: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 載入特定模型設定檔
    model_config_path = Path(f"configs/{model_name}.yaml")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"模型配置文件未找到: {model_config_path}")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # ✨ 使用遞迴更新，讓 model_config 覆蓋 base_config
    config = _recursive_update(config, model_config)

    # === 後續流程完全不變，它們現在操作的是已經合併好的 config ===
    
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['quant_method'] = quant_method.lower()
    config['embedding_modality'] = embedding_modality.lower()
    
    if config['quant_method'] not in VALID_QUANT_METHODS:
        raise ValueError(f"不支持的量化方法: {quant_method}。可选: {VALID_QUANT_METHODS}")

    # 2. 獨立地載入量化設定檔
    quant_config_path = Path(f"../quantization/configs/{quant_method}_config.yaml")
    quant_details = _load_quant_details(quant_config_path, config['quant_method'])
    
    # === 3. 格式化和派生「數據」與「輸出」的路徑 ===
    # 這部分的路徑模板仍然來自 TIGER.yaml，是合理的，因為它定義了數據存放格式
    # 3. 格式化路徑
    paths = config['paths']
    # ✅ 將 model_name 加入字典
    format_args = {
        'dataset_name': dataset_name, 
        'quant_method': config['quant_method'], 
        'model_name': model_name
    }
    dataset_root = Path(paths['dataset_root'].format(**format_args))
    output_root = Path(paths['output_root'].format(**format_args))

        # === 4. 自动构造 codebook 路径 ===
    dataset_root = Path(f"../datasets/{dataset_name}")
    codebook_dir = dataset_root / "codebooks"

    mod_tag = embedding_modality.lower()
    quant_tag = config['quant_method'].lower()

    # 严格匹配指定模态和量化方法
    codebook_path = codebook_dir / f"{dataset_name}.{mod_tag}.{quant_tag}.npy"

    if not codebook_path.exists():
        raise FileNotFoundError(
            f"[FATAL] 未找到指定模态 '{mod_tag}' 的 codebook 文件！\n"
            f"期望路径: {codebook_path}\n"
            f"请确认路径及文件名与保存时一致。"
        )

    config['code_path'] = str(codebook_path)
    logging.info(f"📦 [Config] 成功加载 Codebook: {config['code_path']}")

    config['log_path'] = output_root / "training.log"
    config['save_path'] = output_root / "best_model.pth"
    config['train_json'] = dataset_root / f"{dataset_name}.train.jsonl"
    config['valid_json'] = dataset_root / f"{dataset_name}.valid.jsonl"
    config['test_json'] = dataset_root / f"{dataset_name}.test.jsonl"
    _ensure_dir_exists(output_root)

    # === 4. 根據載入的量化細節，計算詞表參數 ===
    K = int(quant_details['codebook_size'])
    num_semantic_levels = int(quant_details['num_levels'])
    has_dup_layer = quant_details.get('has_dup_layer', True) 
    
    config['codebook_size'] = K
    config['num_semantic_levels'] = num_semantic_levels

    # === 5. 校验 codebook 檔案 ===
    if not Path(config['code_path']).is_file():
        raise FileNotFoundError(f"[FATAL] 未找到 codebook: {config['code_path']}")
    codes_arr = np.load(config['code_path'], allow_pickle=True)
    codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
    
    expected_code_len = num_semantic_levels + 1 if has_dup_layer else num_semantic_levels
    config['code_len'] = expected_code_len

    if codes_mat.ndim != 2 or codes_mat.shape[1] != expected_code_len:
        raise ValueError(f"[FATAL] Codebook {config['code_path']} 的期望形状為 (N, {expected_code_len})，實際為 {codes_mat.shape}")

    # === 6. 計算最終詞表參數 ===
    if has_dup_layer:
        dup_max = int(codes_mat[:, -1].max()) if codes_mat.size > 0 else 0
        dup_vocab_size = dup_max + 1
        config['dup_vocab_size'] = dup_vocab_size
        vocab_sizes = [K] * num_semantic_levels + [dup_vocab_size]
    else:
        vocab_sizes = [K] * num_semantic_levels
    config['vocab_sizes'] = vocab_sizes

    bases = np.cumsum([0] + vocab_sizes[:-1]).tolist()
    config['bases'] = [int(b) for b in bases]

    # === 6. 定義特殊 Token (統一放置在詞表的尾端，避免與 Code 衝突) ===
    base_vocab = sum(config['vocab_sizes'])  # 語義 token 數量
    # 規範：PAD 固定為 0，語義 token 全部偏移 +1，特殊 token 追加在詞表尾部
    pad_id = 0
    mask_id = base_vocab + 1
    cls_id = base_vocab + 2
    sep_id = base_vocab + 3
    eos_id = base_vocab + 4  # 保留 EOS
    vocab_size = eos_id + 1  # ID 范圍 0..eos_id

    config['token_params'] = {
        'pad_token_id': pad_id,
        'cls_token_id': cls_id,
        'sep_token_id': sep_id,
        'mask_token_id': mask_id,
        'eos_token_id': eos_id,
        'vocab_size': vocab_size
    }
    
    return config


# --- 日志和随机种子函数保持不变 ---
def setup_logging(log_path: Path):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def get_model_class(model_name: str):
    """
    增强版：根据模型名称字符串，动态地从 'models' 目录及其所有子目录中
    搜索并导入对应的模型类。

    Args:
        model_name (str): 模型的名称 (e.g., "TIGER")。
                          约定：文件名和类名都应与 model_name 一致 (TIGER.py, class TIGER)。
    """
    models_root_dir = "models"
    model_file_name = f"{model_name}.py"
    model_module_path = None

    # os.walk 会遍历指定目录下的所有文件夹和文件
    for root, dirs, files in os.walk(models_root_dir):
        if model_file_name in files:
            # 找到了文件！现在构建 Python 的导入路径
            # 例如, root = "models/encoder_decoder"
            
            # 1. 将文件系统路径 ('/') 替换为 Python 导入路径 ('.')
            # "models/encoder_decoder" -> "models.encoder_decoder"
            base_path = root.replace(os.sep, '.')
            
            # 2. 拼接成最终的模块路径
            # "models.encoder_decoder.TIGER"
            model_module_path = f"{base_path}.{model_name}"
            break # 找到后立刻停止搜索

    # 如果遍历完都没找到文件
    if not model_module_path:
        raise ImportError(
            f"错误：无法在 '{models_root_dir}' 目录或其任何子目录中找到模型文件 '{model_file_name}'。\n"
            f"请检查你的文件结构和 --model 参数是否正确。"
        )

    try:
        # 使用动态构建的路径来导入模块
        logging.info(f"Found model module at: {model_module_path}")
        model_module = importlib.import_module(model_module_path)
        # 从模块中获取与模型同名的类
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"错误：成功找到文件，但在从 '{model_module_path}' 导入类 '{model_name}' 时失败。\n"
            f"请确保你的 Python 文件内 class 的名称 ({model_name}) 与文件名和 --model 参数完全一致。\n"
            f"原始错误: {e}"
        )

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

def load_item_category_map(
    dataset_root: Path,
    dataset_name: str,
    return_cate_names: bool = False,
    min_items_per_cate: int = 5,
    max_categories: int = 0,
    split_threshold: float = 0.01,  # [修改] 默認閾值降為 1%，讓類別更容易被拆分
    use_composite_keys: bool = True # [修改] 新增參數，是否組合最後兩級 (e.g. "Guitars > Electric")
):
    """
    從預處理生成的 item.json 中提取類別信息，返回 {item_id(1-based): category_id} 映射。
    [修改版特性]:
    - 默認採用「葉子優先」(Leaf-first) 策略，直接取最細粒度的分類。
    - 支持組合鍵 (Parent > Child) 以區分不同大類下的同名子類。
    - 降低拆分閾值，大幅增加類別數量，解決多樣性指標階躍問題。
    """
    item_file = dataset_root / f"{dataset_name}.item.json"
    if not item_file.is_file():
        logging.warning(f"[Diversity] item metadata not found at {item_file}, skip category map.")
        return {}

    try:
        with open(item_file, "r", encoding="utf-8") as f:
            item_meta = json.load(f)
    except Exception as exc:
        logging.warning(f"[Diversity] failed to load item metadata ({item_file}): {exc}")
        return {}

    cate2id: dict = {}
    item_to_cate: dict = {}
    missing = 0
    parsed_categories: dict = {}

    # 1. 解析 Categories 層級
    for item_str, info in item_meta.items():
        categories = info.get("categories")
        tokens = []
        if isinstance(categories, list):
            categories = categories[0] if categories else None
        if isinstance(categories, str) and categories.strip():
            tokens = [c.strip() for c in categories.split(",") if c.strip()]
        parsed_categories[item_str] = tokens

    # 2. 檢測並跳過佔比極高的根類（例如 Amazon 數據集中的 "Musical Instruments"）
    first_level_counter = Counter(tokens[0] for tokens in parsed_categories.values() if tokens)
    dominant_root = None
    total_with_cate = sum(1 for tokens in parsed_categories.values() if tokens)
    
    if first_level_counter and total_with_cate:
        root_candidate, cnt = first_level_counter.most_common(1)[0]
        # [修改] 將閾值從 0.9 降為 0.5，更激進地去除無效的根節點
        if cnt / total_with_cate >= 0.5: 
            dominant_root = root_candidate
            logging.info(
                f"[Diversity] Detected dominant root category '{dominant_root}' "
                f"({cnt}/{total_with_cate}); will skip it when possible."
            )
    root_norm = dominant_root.lower() if dominant_root else None

    # 構建去除根類的層級列表
    parsed_no_root = {}
    for item_str, tokens in parsed_categories.items():
        cleaned = tokens
        if tokens and root_norm and tokens[0].lower() == root_norm:
            cleaned = tokens[1:]
        parsed_no_root[item_str] = cleaned

    # 3. 預備 Codebook 作為備用 (保持原邏輯)
    codebook_prefixes = {}
    if dominant_root or missing > 0: # 稍微放寬條件，讓 missing 的時候也能加載
        codebook_dir = dataset_root / "codebooks"
        preferred = codebook_dir / f"{dataset_name}.text.rqvae.codebook.json"
        codebook_file = preferred if preferred.is_file() else None
        if not codebook_file and codebook_dir.is_dir():
            candidates = sorted(codebook_dir.glob("*.codebook.json"))
            if candidates: codebook_file = candidates[0]
        
        if codebook_file:
            try:
                with open(codebook_file, "r", encoding="utf-8") as f:
                    codebook_data = json.load(f)
                for key, value in codebook_data.items():
                    try:
                        idx = int(key)
                        parts = [p.strip("<>") for p in str(value).split() if p]
                        if len(parts) >= 2:
                            codebook_prefixes[idx] = f"{parts[0]}|{parts[1]}"
                        elif parts:
                            codebook_prefixes[idx] = parts[0]
                    except: continue
                logging.info(f"[Diversity] Loaded codebook prefixes for fallback ({len(codebook_prefixes)} entries).")
            except Exception as exc:
                logging.warning(f"[Diversity] failed to load codebook prefixes: {exc}")

    # 4. 第一輪：確定初始類別 (核心修改：葉子優先)
    item_to_raw_cate = {}
    item_tokens_no_root = {} # 保存 tokens 供後續細分使用

    for item_str, info in item_meta.items():
        cat = None
        tokens = parsed_categories.get(item_str, [])
        tokens_no_root = parsed_no_root.get(item_str, [])
        
        # [修改] 優先策略：直接使用去除根節點後的最深層級
        candidate_tokens = tokens_no_root if tokens_no_root else tokens
        
        if candidate_tokens:
            if use_composite_keys and len(candidate_tokens) >= 2:
                # [策略 A] 組合鍵: "Parent > Child" (區分度最高)
                cat = f"{candidate_tokens[-2]} > {candidate_tokens[-1]}"
            else:
                # [策略 B] 葉子節點: 取最後一級
                cat = candidate_tokens[-1]

        # 降級策略 1: Genre (通常比 Brand 寬泛，但也可用)
        if not cat:
            genres = info.get("genres")
            if isinstance(genres, list) and len(genres) > 0:
                cat = str(genres[0]).strip()

        # 降級策略 2: Brand (作為弱類別補充)
        if not cat:
            brand = info.get("brand", "").strip()
            if brand:
                cat = f"Brand: {brand}"

        # 降級策略 3: Codebook
        try:
            iid = int(item_str)
        except ValueError:
            missing += 1
            continue

        if not cat and iid in codebook_prefixes:
            cat = f"Code: {codebook_prefixes[iid]}"

        if not cat:
            missing += 1
            continue

        # 模型內部 item_id 是 1-based，這裡要做轉換
        item_to_raw_cate[iid + 1] = cat
        item_tokens_no_root[iid + 1] = candidate_tokens

    total_items_with_cate = len(item_to_raw_cate)

    # 5. 第二輪：對過於龐大的類別進行強制細分 (使用 split_threshold)
    if total_items_with_cate > 0:
        updated_global = True
        loop_cnt = 0
        # 允許迭代 3 次，應對層級很深的情況
        while updated_global and loop_cnt < 3: 
            updated_global = False
            loop_cnt += 1
            raw_counter = Counter(item_to_raw_cate.values())
            
            for cate, cnt in list(raw_counter.items()):
                # [修改] 使用傳入的低閾值 (e.g. 0.01)
                if cnt / total_items_with_cate <= split_threshold:
                    continue
                
                # 對該類別進行細分嘗試
                for iid, cur_cate in list(item_to_raw_cate.items()):
                    if cur_cate != cate: continue
                    
                    toks = item_tokens_no_root.get(iid, [])
                    if not toks: continue

                    # 嘗試拼接更多層級來區分
                    # 邏輯：如果當前已經用了 "A > B"，嘗試變成 "A > B > C" (如果 C 存在)
                    # 這裡簡化處理：直接取最後 3 級
                    new_cate = None
                    if len(toks) >= 3:
                         new_cate = " > ".join(toks[-3:])
                    elif len(toks) == 2 and cur_cate != " > ".join(toks):
                         new_cate = " > ".join(toks)
                    elif len(toks) == 1 and cur_cate != toks[0]:
                         new_cate = toks[0]
                    
                    if new_cate and new_cate != cate:
                        item_to_raw_cate[iid] = new_cate
                        updated_global = True

    # 6. 第三輪：合併過小類別 (Tail Merging)
    raw_counter = Counter(item_to_raw_cate.values())
    small_cates = {c for c, cnt in raw_counter.items() if cnt < min_items_per_cate}
    
    if max_categories and len(raw_counter) > max_categories:
        for idx, (cate, _) in enumerate(raw_counter.most_common()):
            if idx >= max_categories:
                small_cates.add(cate)
                
    if small_cates:
        other_name = "other"
        for iid, cate in list(item_to_raw_cate.items()):
            if cate in small_cates:
                item_to_raw_cate[iid] = other_name
        raw_counter = Counter(item_to_raw_cate.values())

    # 7. 重新編碼 cate_id
    # 按數量降序排列 ID，方便後續觀察
    sorted_cates = sorted(raw_counter.keys(), key=lambda x: raw_counter[x], reverse=True)
    for cate in sorted_cates:
        cate2id[cate] = len(cate2id)
        
    for iid, cate in item_to_raw_cate.items():
        item_to_cate[iid] = cate2id[cate]

    logging.info(
        f"[Diversity] Loaded fine-grained categories for {len(item_to_cate)} items "
        f"(distinct={len(cate2id)}, missing={missing}, split_thre={split_threshold:.2%})."
    )
    
    if return_cate_names:
        id_to_cate = {cid: cname for cname, cid in cate2id.items()}
        return item_to_cate, id_to_cate
    return item_to_cate