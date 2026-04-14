# 🧩 UniGenRec 数据处理与Embedding

本项目提供从 **原始数据下载 → 数据预处理 → 文本与图像 Embedding 生成 → 多模态融合** 的一站式处理脚本。  
以 Amazon 与 MovieLens 为例。


## 📦 1. 下载数据集

从公开源下载 Amazon 或 MovieLens 数据集：

```bash
# Amazon 数据集
python download_data.py --source amazon --dataset Musical_Instruments

# MovieLens 数据集
python download_data.py --source movielens --dataset ml-1m

# recbole 数据集
python download_recbole_data.py --dataset amazon-musical-instruments-23
```


## 🖼️ 2. 下载图片资源

若数据包含图像内容，可运行以下命令下载对应图片：

```bash
# Amazon 类数据集
python download_images.py --dataset_type amazon --dataset Musical_Instruments

# MovieLens 数据集
python download_images.py --dataset_type movielens --dataset ml-1m
```



## 🧹 3. 数据预处理

对原始数据执行清洗、格式化与标准化：

```bash
# Amazon
python process_data.py --dataset_type amazon --dataset Musical_Instruments

# MovieLens
python process_data.py --dataset_type movielens --dataset ml-1m


python process_data.py --dataset_type recbole --dataset ml-100k
```

---

## 🔠 4. Embedding 生成

### 生成本地 T5 文本嵌入 (PCA 到 512d):

```bash
python process_embedding.py  --embedding_type text_local --dataset ml-100k  --model_name_or_path sentence-transformers/sentence-t5-base --pca_dim 512
```

### 生成 OpenAI API 文本嵌入:

```bash
python process_embedding.py \
    --embedding_type text_api \
    --dataset Books \
    --sent_emb_model text-embedding-3-large \
    --pca_dim 512
```

### 生成 CLIP 图像嵌入:


```bash
python process_embedding.py \
    --embedding_type image_clip \
    --dataset Musical_Instruments \
    --clip_model_name /home/peiyu/PEIYU/LLM_Models/openai-mirror/clip-vit-base-patch32 \
    --pca_dim 512
```

### 生成 SASRec 协同嵌入:

```bash
python process_embedding.py \
    --embedding_type cf_sasrec \
    --dataset Sports_and_Outdoors \
    --sasrec_hidden_dim 64 \
    --sasrec_epochs 100 \
    --pca_dim 0
```

## 5. 模态融合

```bash
python preprocessing/train_fusion_attention.py \
  --dataset Baby \
  --text_model_tag "Qwen/Qwen3-VL-7B-Instruct" \
  --image_model_tag "openai/clip-vit-base-patch32" \
  --fusion_out_dim 512 \
  --epochs 10 \
  --batch_size 1024 \
  --output_tag "sota-attn"
```