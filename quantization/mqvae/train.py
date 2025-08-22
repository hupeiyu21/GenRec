# /quantization/mqvae/train.py

import os
import sys
import json
import logging
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从同级目录导入模型定义和通用工具
from .mqvae import MQVAE_Rec
from quantization import utils

def _train_epoch(model, dataloader, optimizer, beta, device, is_eval=False):
    """
    单个 epoch 的训练/评估，修复 shape 不匹配问题。
    """
    model.eval() if is_eval else model.train()
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0
    
    for (x_batch,) in dataloader:
        x_batch = x_batch.to(device)
        if not is_eval: 
            optimizer.zero_grad()
        
        # squeeze 多余维度，保证 [B, D]，消除广播警告
        x_batch_flat = x_batch.squeeze(1) if x_batch.dim() == 3 and x_batch.shape[1] == 1 else x_batch

        with torch.set_grad_enabled(not is_eval):
            recon_x, commitment_loss, _ = model(x_batch_flat)
            # 同样 squeeze recon_x 如果模型输出多余维度
            recon_x_flat = recon_x.squeeze(1) if recon_x.dim() == 3 and recon_x.shape[1] == 1 else recon_x

            reconstruction_loss = F.mse_loss(recon_x_flat, x_batch_flat, reduction="mean")
            # 如果 commitment_loss 是 None，需要处理
            commitment_loss_value = commitment_loss if commitment_loss is not None else 0.0
            loss = reconstruction_loss + beta * commitment_loss_value
        
        if not is_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss += loss.item()
        total_rec_loss += reconstruction_loss.item()
        total_commit_loss += commitment_loss_value if commitment_loss is not None else 0.0
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_rec_loss / num_batches, total_commit_loss / num_batches


def run_pipeline(embeddings, device, config, ckpt_dir, codebook_dir):
    """
    这个函数将被顶层的 main.py 调用。
    """
    logging.info("================== MQ-VAE Pipeline-开始 ==================")
    
    # 1. 初始化模型
    input_dim = embeddings.shape[1]
    model = MQVAE_Rec(
        input_dim=input_dim,
        **config['model_params']
    ).to(device)
    logging.info("MQ-VAE 模型初始化完成。")

    # 2. 训练模型
    logging.info("开始训练 MQ-VAE 模型...")
    train_cfg = config['training_params']
    batch_size, num_epochs, lr, beta = train_cfg['batch_size'], train_cfg['epochs'], train_cfg['lr'], train_cfg['beta']
    optimizer = getattr(torch.optim, train_cfg['optimizer'])(model.parameters(), lr=lr, weight_decay=train_cfg.get('weight_decay', 0.0))
    
    train_data, val_data = train_test_split(embeddings, test_size=0.05, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.Tensor(train_data)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(val_data)), batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    for epoch in tqdm(range(num_epochs), desc="训练 MQ-VAE"):
        _train_epoch(model, train_loader, optimizer, beta, device)
        if (epoch + 1) % 100 == 0:
            val_loss, val_rec, val_commit = _train_epoch(model, val_loader, None, beta, device, is_eval=True)
            logging.info(f"[VAL] Ep {epoch+1:04d} | Loss: {val_loss:.4f} (Rec: {val_rec:.4f}, Commit: {val_commit:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"模型已保存到: {best_model_path}")

    logging.info("训练完成。")

    # 3. 加载最佳模型
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行码本生成: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")
    model.eval()

    # 4. 生成码本
    logging.info("开始生成码本...")
    model_cfg = config['model_params']
    vocab_size = model_cfg['codebook_size']

    all_codes = []
    dataloader = DataLoader(TensorDataset(torch.tensor(embeddings, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (x_batch,) in tqdm(dataloader, desc="生成基础码"):
            codes = model.get_codes(x_batch.to(device)).detach().cpu().numpy().astype(np.int64)
            all_codes.append(codes)

    base_sids_np = np.vstack(all_codes)

    dedup_layer = utils.build_dedup_layer(base_sids_np, vocab_size)
    final_codes_np = np.concatenate([base_sids_np, dedup_layer], axis=1).astype(np.int32)
    logging.info(f"最终码本维度: {final_codes_np.shape}")
    
    # 保存码本
    json_path = os.path.join(codebook_dir, "mqvae_codebook.json")
    with open(json_path, 'w') as f:
        json.dump({str(i): final_codes_np[i].tolist() for i in range(len(final_codes_np))}, f, indent=2)
    pt_path = os.path.join(codebook_dir, "mqvae_codebook.pt")
    torch.save(torch.from_numpy(final_codes_np.T).contiguous().long(), pt_path)
    logging.info(f"码本已保存: JSON -> {json_path}, PT -> {pt_path}")