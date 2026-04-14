# /quantization/main.py

import argparse
import numpy as np
import torch
import logging
import os
import sys
import importlib

# 添加项目根目录到Python路径 (如果 utils 在当前目录，这行可能不需要)
# sys.path.append(os.path.abspath(os.path.dirname(__file__))) 

import utils
# 假设 Trainer 已更新以处理多模态
from trainer import Trainer 
# (如果 MultiModalDataset 在 dataset.py)
# from dataset import MultiModalDataset 

def main():
    parser = argparse.ArgumentParser(description="通用量化器训练脚本")
    
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['rqvae', 'vqvae', 'rkmeans','rvq', 'mqvae','opq', 'pq', 'rqvae_letter', 'mm_rqvae'], 
                        help='要使用的量化器模型名称。')
                        
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (e.g., Baby)')
    
    # --- ✅ 修改：拆分 embedding_model 为 text 和 image (可选) ---
    parser.add_argument('--embedding_modality', type=str, default='text', 
                        help="[单模态] 嵌入类型 (e.g., 'text', 'image')。")
                        
    parser.add_argument('--embedding_model', type=str, default=None, # 改为可选
                        help='[单模态] 嵌入来源模型名称 (e.g., sentence-t5-base)。')

    parser.add_argument('--text_embedding_model', type=str, default=None,
                        help="[仅用于 MM_RQVAE] 文本嵌入的来源模型名称。")
    parser.add_argument('--image_embedding_model', type=str, default=None,
                        help="[仅用于 MM_RQVAE] 图像或融合嵌入的来源模型名称。")
                        
    # --- 移除显式路径 ---
    # parser.add_argument('--text_embedding_path', ...)
    # parser.add_argument('--image_embedding_path', ...)
                        
    # (其他参数保持不变)
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径。')
    parser.add_argument('--data_base_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--log_base_path', type=str, default='../logs/quantization', help='日志根目录')
    parser.add_argument('--ckpt_base_path', type=str, default='../ckpt/quantization', help='模型根目录')
    parser.add_argument('--codebook_base_path', type=str, default='../datasets', help='码本根目录')
    
    args = parser.parse_args()

    # --- 1. 设置路径和日志 ---
    try:
        # setup_paths 现在会根据 model_name 和 text/image_embedding_model 处理路径
        embedding_path, log_dir, ckpt_dir, codebook_base_dir = utils.setup_paths(args)
    except ValueError as e: # setup_paths 现在可能抛出 ValueError
        logging.error(f"路径设置失败: {e}"); return
        
    utils.setup_logging(log_dir)

    # --- 2. 加载配置 ---
    logging.info(f"任务: model={args.model_name}, dataset={args.dataset_name}")
    # (自动路由逻辑)
    if args.config_path is None: config_path = f"./configs/{args.model_name}_config.yaml"
    else: config_path = args.config_path
    logging.info(f"使用配置文件: {config_path}")
    if not os.path.exists(config_path): logging.error(f"配置文件未找到: {config_path}"); return
    config = utils.load_yaml_file(config_path)
    config['model_name'] = args.model_name
    config['dataset_name'] = args.dataset_name
    
    # --- 3. 加载数据 ---
    is_multimodal = isinstance(embedding_path, tuple)
    embeddings_data = None
    input_size_text = 0
    input_size_image = 0
    input_size_single = 0
    actual_embedding_path_info = "" 

    try:
        if is_multimodal:
            path_T, path_I = embedding_path
            logging.info(f"加载文本特征: {path_T}")
            item_embeddings_T = np.load(path_T)
            logging.info(f"加载图像特征: {path_I}")
            item_embeddings_I = np.load(path_I)
            
            if item_embeddings_T.shape[0] != item_embeddings_I.shape[0]:
                raise ValueError("文本和图像特征数量不匹配!")
                
            logging.info(f"文本特征维度: {item_embeddings_T.shape}")
            logging.info(f"图像特征维度: {item_embeddings_I.shape}")
            
            config['total_item_count'] = item_embeddings_T.shape[0]
            embeddings_data = (item_embeddings_T, item_embeddings_I)
            input_size_text = item_embeddings_T.shape[1]
            input_size_image = item_embeddings_I.shape[1]
            actual_embedding_path_info = f"\n  Text: {path_T}\n  Image/Fused: {path_I}"
            
        else: # 单模态
            logging.info(f"加载特征文件: {embedding_path}")
            item_embeddings = np.load(embedding_path)
            logging.info(f"特征加载完成, 维度: {item_embeddings.shape}")
            config['total_item_count'] = len(item_embeddings)
            embeddings_data = item_embeddings
            input_size_single = item_embeddings.shape[1]
            actual_embedding_path_info = f": {embedding_path}"

    except FileNotFoundError as e: logging.error(f"特征文件未找到: {e}"); return
    except ValueError as e: logging.error(f"加载数据时出错: {e}"); return
    except Exception as e: logging.error(f"加载数据时发生未知错误: {e}"); return
         
    logging.info(f"最终使用的嵌入文件{actual_embedding_path_info}") 
    
    device = torch.device(config['common'].get('device', 'cuda:0'))
    logging.info(f"使用设备: {device}")
    
    # --- 4. 初始化模型 ---
    logging.info(f"正在加载模型: {args.model_name}...")
    try:
        ModelClass = utils.get_model(args.model_name)
        logging.info(f"成功加载模型类: {ModelClass.__name__}")
        
        if is_multimodal:
            model = ModelClass(config=config, input_size_text=input_size_text, input_size_image=input_size_image).to(device)
        else:
            model = ModelClass(config=config, input_size=input_size_single).to(device)
            
    except ValueError as e: logging.error(f"初始化模型失败: {e}"); return
    except Exception as e: logging.error(f"初始化模型时发生未知错误: {e}"); return

    # --- 5. 执行 Trainer ---
    try:
        trainer = Trainer(config=config, model=model, device=device)
        
        best_model_path = trainer.fit(
            embeddings_data=embeddings_data, 
            ckpt_dir=ckpt_dir
        )
        
        # (加载最佳模型)
        if best_model_path and os.path.exists(best_model_path): # 检查 best_model_path 是否有效
             logging.info(f"加载最佳模型进行码本生成: {best_model_path}")
             try: model.load_state_dict(torch.load(best_model_path, map_location=device))
             except Exception as e: logging.warning(f"加载最佳模型失败 ({e})。使用最终模型。")
        else:
             logging.warning(f"未找到或无效的最佳模型路径: {best_model_path}。使用训练结束时的模型。")

        
        # ✅ (修改) 调用 build_codebook_path 时传递正确的参数
        final_codebook_path = utils.build_codebook_path(
            codebook_base_path=args.codebook_base_path, 
            dataset_name=args.dataset_name, 
            model_name=args.model_name, 
            # --- 传递正确的模型名称 ---
            text_embedding_model=args.text_embedding_model if is_multimodal else None,
            image_embedding_model=args.image_embedding_model if is_multimodal else None,
            embedding_model=args.embedding_model if not is_multimodal else None,
            # --- 传递单模态的模态名称 ---
            embedding_modality=args.embedding_modality if not is_multimodal else None 
        )
        logging.info(f"最终码本将保存到: {final_codebook_path}")
        
        trainer.predict(
            embeddings_data=embeddings_data, 
            output_path=final_codebook_path 
        )
        
    except Exception as e:
        logging.error(f"Trainer 执行过程中出错: {e}", exc_info=True); return

    
    logging.info("\n--- 所有任务完成 ---")


if __name__ == '__main__':
    main()
