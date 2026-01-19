import argparse
import logging
import torch
import torch.optim as optim
import os
import pprint
from collections import Counter
from typing import Optional # ✅ (新增) 导入 Optional

# ✅ 1. 從 torch.utils.data 直接導入 DataLoader
from torch.utils.data import DataLoader 
from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer       
from trainer import train_one_epoch, evaluate
from utils import (
    load_and_process_config,
    setup_logging,
    set_seed,
    get_model_class,
    load_item_category_map,
)

import sys
from pathlib import Path

# 获取项目根目录路径（main.py 的上一级目录的上一级）
root = Path(__file__).resolve().parent  # recommendation/
root_parent = root.parent               # 项目根目录

if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

from recommendation.models.generation.prefix_tree import Trie, build_trie_from_codebook


def main():
    # === 1. 解析命令列參數 ===
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str,default="ReGen", help='模型名稱 (e.g., TIGER, GPT2, RPG)')
    parser.add_argument('--dataset', type=str, default="amazon-video-games-23", help='数据集名稱 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, default="rqvae", choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq', 'vqvae', 'mm_rqvae'], help='量化方法')
    parser.add_argument('--embedding_modality', type=str, default='text', choices=['text', 'image', 'fused', 'lfused', 'cf'], help='量化模态类型，对应不同的 codebook (默认 text)')
    parser.add_argument('--eval_only', default=True, help='仅加载已有模型，在测试集上直接评估')

    
    # ✅ (已移除) 删除了 --no_trie 命令行参数
    
    args = parser.parse_args()
    eval_only = args.eval_only


    # === 2. 載入並處理設定檔 ===
    config = load_and_process_config(
        args.model, 
        args.dataset, 
        args.quant_method,
        embedding_modality=args.embedding_modality
    )
    ckpt_override = config['save_path']
    print(f"ckpt_override: {ckpt_override}") 

    # === 3. 初始化 (日誌, 隨機種子) ===
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")
    logging.info("=" * 50)
    config_str = pprint.pformat(config)
    logging.info("\n" + config_str)
    logging.info("=" * 50)

    # === 4. 設定設備 ===
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_workers = config['training_params'].get('num_workers', 4)
    
    # === 5. ✅ (顺序调整) 载入 item_to_code 映射 ===
    # (必须在创建模型之前完成，因为Trie依赖它)
    logging.info("Loading item to code mapping...")
    item_to_code_map, code_to_item_map = item2code(
        config['code_path'],
        config['vocab_sizes'],
        config['bases']
    )
    logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")
    dataset_root = Path(config['train_json']).parent
    # === 修改调用参数 ===
    item_to_cate_map, cate_id_to_name = load_item_category_map(
        dataset_root,
        args.dataset,
        return_cate_names=True,
        min_items_per_cate=int(config['evaluation_params'].get('diversity_min_items_per_cate', 5)), # 读取配置
        max_categories=int(config['evaluation_params'].get('diversity_max_categories', 0)),         # 读取配置
        split_threshold=float(config['evaluation_params'].get('diversity_split_threshold', 0.01)),
        use_composite_keys=bool(config['evaluation_params'].get('diversity_use_composite_keys', True))
    )
    if item_to_cate_map:
        cate_counter = Counter(item_to_cate_map.values())
        total = sum(cate_counter.values())
        lines = []
        for cid, cnt in cate_counter.most_common():
            name = cate_id_to_name.get(cid, str(cid))
            ratio = (cnt / total) if total else 0
            lines.append(f"{name}: {cnt} ({ratio:.2%})")
        logging.debug("[Diversity] Category distribution:\n" + "\n".join(lines))
    else:
        logging.debug("[Diversity] Category map is empty; skip distribution logging.")

    # === 6. ✅ (修改) 根据 config 构建前缀树 ===
    prefix_trie: Optional[Trie] = None
    
    # 检查 config['evaluation_params'] 中的 'use_prefix_trie' 标志
    # 默认值为 False (如果您希望默认不使用)
    use_trie = config.get('evaluation_params', {}).get('use_prefix_trie', False) 
    
    if use_trie and build_trie_from_codebook is not None:
        logging.info("Building Prefix Trie (enabled in config)...")
        
        # 获取所有合法的 code token 序列
        all_token_sequences = list(item_to_code_map.values())
        
        # 获取 EOS token ID
        eos_token_id = config['token_params']['eos_token_id']
        
        prefix_trie = build_trie_from_codebook(
            token_sequences=all_token_sequences,
            eos_token_id=eos_token_id
        )
    elif use_trie:
        logging.warning("Config requested Prefix Trie, but 'utils.prefix_trie' module was not found.")
    else:
        logging.info("Prefix Trie is DISABLED (default or as per config).")


    # === 7. ✅ (修改) 创建模型 (将Trie注入) ===
    logging.info(f"Dynamically loading model: {args.model}")
    ModelClass = get_model_class(args.model)
    
    # ✅ (修改) 将 config 和 prefix_trie (可能是 None) 传递给模型
    #    (我们假设 ModelClass 的 __init__ 接受 prefix_trie=None)
    model_kwargs = {"prefix_trie": prefix_trie}
    if args.model.upper() == "ReGen" or args.model.upper() == "TIGER_MMR":
        model_kwargs.update(
            {
                "item_to_code_map": item_to_code_map,
                "code_to_item_map": code_to_item_map,
                "item_to_cate_map": item_to_cate_map,
            }
        )
    model = ModelClass(config, **model_kwargs) 
    
    model.to(device)
    logging.info(model.n_parameters)
    logging.info("=" * 50)
    weight_decay = float(config['training_params'].get('weight_decay', 0.01))
    optimizer = None
    if not eval_only:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['training_params']['lr']),
            weight_decay=weight_decay
        )

    # === 8. (顺序调整) 初始化模型专属的 Tokenizer ===
    logging.info(f"Initializing tokenizer for model: {args.model}")
    tokenizer_collate_fn = get_tokenizer(
        model_name=args.model,
        config=config,
        item_to_code_map=item_to_code_map
    )
    logging.info("Tokenizer initialized.")

    # 支持部分模型區分訓練/評估兩種 tokenizer
    if isinstance(tokenizer_collate_fn, dict):
        train_collate_fn = tokenizer_collate_fn.get('train')
        eval_collate_fn = tokenizer_collate_fn.get('eval', train_collate_fn)
    else:
        train_collate_fn = tokenizer_collate_fn
        eval_collate_fn = tokenizer_collate_fn

    # === 9. (顺序调整) 創建數據集與 DataLoader ===
    logging.info("Creating Datasets...")
    if eval_only:
        test_dataset = GenRecDataset(config=config, mode='test')
        print(f"test_dataset len : {len(test_dataset)}")
    else:
        train_dataset = GenRecDataset(config=config, mode='train')
        validation_dataset = GenRecDataset(config=config, mode='valid')
        test_dataset = GenRecDataset(config=config, mode='test')

    logging.info("Creating DataLoaders...")
    
    is_gpu_training = (torch.cuda.is_available() and num_workers > 0)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": train_collate_fn, # 默認使用訓練 tokenizer
        "pin_memory": is_gpu_training,
        "persistent_workers": is_gpu_training if num_workers > 0 else False
    }

    if eval_only:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation_params']['batch_size'],
            shuffle=False, 
            collate_fn=eval_collate_fn,
            num_workers=num_workers,
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training if num_workers > 0 else False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training_params']['batch_size'],
            shuffle=True, 
            **loader_kwargs
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config['evaluation_params']['batch_size'],
            shuffle=False, 
            collate_fn=eval_collate_fn,
            num_workers=num_workers,
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training if num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation_params']['batch_size'],
            shuffle=False, 
            collate_fn=eval_collate_fn,
            num_workers=num_workers,
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training if num_workers > 0 else False
        )

    # === 10. Eval-Only 快捷路径 ===
    if eval_only:
        ckpt_path = Path(ckpt_override) if ckpt_override else Path(config['save_path'])
        if not ckpt_path.is_file():
            logging.error(f"[Eval-Only] Checkpoint not found: {ckpt_path}")
            return
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"[Eval-Only] Loaded checkpoint from {ckpt_path}")

        test_results = evaluate(
            model,
            test_loader,
            config['evaluation_params']['topk_list'],
            device
        )
        logging.info(f"[Eval-Only] Test Results: {test_results}")
        
        return

    # === 11. (顺序调整) 训练-评估循环 (已修改) ===
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    
    eval_interval = config['training_params'].get('eval_interval', 1) 
    logging.info(f"Evaluation interval set to: {eval_interval} epoch(s)")

    for epoch in range(config['training_params']['num_epochs']):
        epoch_num = epoch + 1 
        logging.info(f"--- Epoch {epoch_num}/{config['training_params']['num_epochs']} ---")
        
        # --- 训练 ---
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        # --- 评估 (根据 eval_interval) ---
        if epoch_num % eval_interval == 0:
            logging.info(f"--- Evaluating at Epoch {epoch_num} ---")
            val_results = evaluate(
                model,
                validation_loader,
                config['evaluation_params']['topk_list'],
                device
            )
            logging.info(f"Validation Results: {val_results}")

            current_ndcg = val_results.get('NDCG@10', val_results.get('NDCG@20', 0.0))

            # --- 检查性能提升和 Early Stopping ---
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                early_stop_counter = 0 
                logging.info(f"🚀 New best NDCG on validation: {best_ndcg:.4f} at Epoch {epoch_num}")

                test_results = evaluate(
                    model,
                    test_loader,
                    config['evaluation_params']['topk_list'],
                    device
                )
                logging.info(f"Test Results: {test_results}")

                best_epoch = epoch_num
                best_val_results = val_results
                best_test_results = test_results

                torch.save(model.state_dict(), config['save_path'])
                logging.info(f"Best model saved to {config['save_path']}")
            
            else:
                early_stop_counter += eval_interval 
                logging.info(f"No improvement since Epoch {best_epoch}. Early stop counter: {early_stop_counter}/{config['training_params']['early_stop'] * eval_interval}")
                if early_stop_counter >= config['training_params']['early_stop'] * eval_interval:
                    logging.info("Early stopping triggered.")
                    break
        else:
            logging.info(f"Skipping evaluation for Epoch {epoch_num}.")

    # === 12. (顺序调整) 訓練結束總結 ===
    logging.info("="*50)
    logging.info("🏁 Training Finished!")
    if best_test_results:
        logging.info(f"🏆 Best performance found at Epoch {best_epoch}")
        logging.info(f"  - Best Validation Results: {best_val_results}")
        logging.info(f"  - Corresponding Test Results: {best_test_results}")
    else:
        logging.info("No improvement was observed.")
    logging.info("="*50)


if __name__ == "__main__":
    main()
