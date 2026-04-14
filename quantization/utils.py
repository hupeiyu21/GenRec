# /quantization/utils.py

import os
import sys 
import logging
import importlib 
from datetime import datetime
from collections import defaultdict
import numpy as np
import yaml

def get_model(model_name: str):
    """
    模型工厂函数，支持 MM_RQVAE。
    """
    try:
        module_path = f'models.{model_name}'
        model_module = importlib.import_module(module_path)
        
        class_name = model_name.upper()
        if model_name.lower() == 'mm_rqvae':
             class_name = 'MM_RQVAE' # 确保匹配您文件中的类名
             
        model_class = getattr(model_module, class_name)
        
    except (ImportError, AttributeError) as e:
        print(f"ERROR: 尝试加载模型 '{model_name}' 时失败。 异常: {e}")
        class_name_upper = model_name.upper() 
        if model_name.lower() == 'mm_rqvae': class_name_upper = 'MM_RQVAE'
        
        raise ValueError(
            f'Model "{model_name}" not found. '
            f'请检查:\n'
            f'1. "models/" 中是否存在 "{model_name}.py"。\n'
            f'2. 该文件中是否定义了类 "{class_name_upper}"。'
        )
        
    return model_class


def load_yaml_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def setup_paths(args):
    """根据输入参数构建路径 (自动处理单模态和多模态)"""
    emb_dir = os.path.join(args.data_base_path, args.dataset_name, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    is_multimodal = args.model_name.lower() == 'mm_rqvae'
    embedding_path = None 
    output_base_dir = "" 

    if is_multimodal:
        # --- 多模态路径逻辑 ---
        print("[INFO] 检测到 MM_RQVAE，将根据 text/image embedding model 名称查找文件...")
        
        # 检查必需的参数
        if not args.text_embedding_model or not args.image_embedding_model:
            raise ValueError("错误：当使用 '--model_name mm_rqvae' 时，必须同时提供 '--text_embedding_model' 和 '--image_embedding_model' 参数。")

        text_modality_name = 'text'
        image_modality_name = 'image' # 或者 'fused'，取决于您的文件名约定
        
        # 使用指定的模型名称构建路径
        embedding_filename_T = f"{args.dataset_name}.emb-{text_modality_name}-{args.text_embedding_model}.npy"
        embedding_path_T = os.path.join(emb_dir, embedding_filename_T)
        
        embedding_filename_I = f"{args.dataset_name}.emb-{image_modality_name}-{args.image_embedding_model}.npy"
        # 尝试 fused 命名（如果 image 不存在）
        embedding_path_I_alt = os.path.join(emb_dir, f"{args.dataset_name}.emb-fused-{args.image_embedding_model}.npy")
        embedding_path_I = os.path.join(emb_dir, embedding_filename_I)
        
        # 检查图像/融合文件是否存在
        if not os.path.exists(embedding_path_I):
             if os.path.exists(embedding_path_I_alt):
                  embedding_path_I = embedding_path_I_alt
                  print(f"[INFO] 未找到 'emb-image-...', 使用 'emb-fused-...': {embedding_path_I}")
             # else: 留下原始路径，让后面的加载逻辑报错

        # 返回路径元组
        embedding_path = (embedding_path_T, embedding_path_I)
        
        # 输出目录：包含两个来源模型，更清晰
        output_base_dir = f"{args.model_name}/{args.text_embedding_model}+{args.image_embedding_model}" 

    else:
        # --- 单模态路径逻辑 ---
        if not args.embedding_modality:
             print("[WARN] 未指定 embedding_modality，默认为 'text'。")
             args.embedding_modality = 'text'
        if not args.embedding_model:
             raise ValueError("错误：对于单模态模型，必须提供 '--embedding_model' 参数。")
             
        embedding_filename = f"{args.dataset_name}.emb-{args.embedding_modality}-{args.embedding_model}.npy"
        embedding_path = os.path.join(emb_dir, embedding_filename)
        
        # 输出目录
        output_base_dir = f"{args.model_name}/{args.embedding_modality}-{args.embedding_model}"

    # --- 共享的输出路径构建 ---
    log_dir = os.path.join(args.log_base_path, args.dataset_name, output_base_dir)
    ckpt_dir = os.path.join(args.ckpt_base_path, args.dataset_name, output_base_dir)
    codebook_base_dir = os.path.join(args.codebook_base_path, args.dataset_name, "codebooks")

    for d in [log_dir, ckpt_dir, codebook_base_dir]:
        os.makedirs(d, exist_ok=True)

    print("--- 自动构建路径 ---")
    if is_multimodal:
        print(f"输入嵌入 (Text): {embedding_path[0]}")
        print(f"输入嵌入 (Image/Fused): {embedding_path[1]}")
    else:
        print(f"输入嵌入文件: {embedding_path}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {ckpt_dir}")
    print(f"码本根目录: {codebook_base_dir}")
    print("------------------------------------\n")

    return embedding_path, log_dir, ckpt_dir, codebook_base_dir

def setup_logging(log_dir):
    """配置日志记录器"""
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging setup complete.")

def build_dedup_layer(base_codes_np: np.ndarray, vocab_size: int):
    """
    为基础码本添加一个去重层。
    对基础码完全相同的条目，在各自簇内部分配 0..k-1 的ID。
    这是一个通用逻辑，可以被任何产生分层码本的模型复用。
    """
    logging.info("构建去重层...")
    N = base_codes_np.shape[0]
    groups = defaultdict(list)
    for idx, key in enumerate(map(tuple, base_codes_np)):
        groups[key].append(idx)

    dedup_layer = np.zeros((N, 1), dtype=np.int64)
    max_dup, overflow_count = 0, 0
    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab_size:
            logging.warning(f"一个簇内重复数 {k} > 码本大小 {vocab_size}。去重ID将取模，可能导致碰撞。")
            local_ids = np.arange(k, dtype=np.int64) % vocab_size
            overflow_count += 1
        else:
            local_ids = np.arange(k, dtype=np.int64)
        dedup_layer[np.array(idx_list), 0] = local_ids
    
    logging.info(f"去重层构建完成。最大簇内重复数: {max_dup}。发生取模的簇数量: {overflow_count}。")
    return dedup_layer

def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        data = data[:, 0, :]
    ids = model.get_codes(data).cpu().numpy()
    max_item_calculate = 1000
    cos_sim_array = np.zeros(config["num_levels"])

    for n_prefix in range(1, config["num_levels"] + 1):
        unique_prefix = np.unique(ids[:, :n_prefix], axis=0)
        this_level_cos_sim_within_cluster = []

        for this_level_prefix in unique_prefix:
            mask = (ids[:, :n_prefix] == this_level_prefix).all(axis=1)
            this_cluster = data[mask].cpu()
            this_cluster_num = this_cluster.shape[0]

            if this_cluster_num > 1:
                indice = torch.randperm(this_cluster_num)[:max_item_calculate]
                cos_sim = F.cosine_similarity(
                    this_cluster[indice, :, None],
                    this_cluster.t()[None, :, indice]
                )
                cos_sim_sum = torch.tril(cos_sim, diagonal=-1).sum()
                normalization_factor = (this_cluster_num - 1) * this_cluster_num / 2
                this_level_cos_sim_within_cluster.append(
                    cos_sim_sum.item() / normalization_factor
                )

        if this_level_cos_sim_within_cluster:
            cos_sim_array[n_prefix - 1] = np.mean(this_level_cos_sim_within_cluster)

    return cos_sim_array


def process_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    final_output_path = os.path.join("cache", type, category, "processed", "final_pca_embeddings.npy")

    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Embedding file not found: {final_output_path}")

    np_array = np.load(final_output_path)
    tensor = torch.from_numpy(np_array).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Loaded embeddings from '{final_output_path}', shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay

def build_codebook_path(codebook_base_path: str, dataset_name: str, 
                        model_name: str, 
                        text_embedding_model: str = None, 
                        image_embedding_model: str = None, 
                        embedding_model: str = None,
                        embedding_modality: str = None) -> str:
    """
    构建码本路径（不再包含具体 embedding 模型名，仅保留模态和模型类型）
    例如：
        单模态: Baby.fused.rqvae.npy
        多模态: Baby.text+image.mm_rqvae.npy
    """
    ds = str(dataset_name)
    model_tag = str(model_name).lower()
    dir_path = os.path.join(codebook_base_path, ds, "codebooks")
    os.makedirs(dir_path, exist_ok=True) 

    # --- 多模态情况 ---
    if model_tag == 'mm_rqvae':
        filename = f"{ds}.lfused.{model_tag}.npy"

    # --- 单模态情况 ---
    else:
        mod_tag = str(embedding_modality).lower() if embedding_modality else 'text'
        filename = f"{ds}.{mod_tag}.{model_tag}.npy"

    return os.path.join(dir_path, filename)
