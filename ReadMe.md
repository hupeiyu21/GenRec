<div align="center">
  <h1>ReGen: Controllable Generative Recommendation via Adaptive Semantic Guidance in Discrete Diffusion</h1>
</div>

---

# Pipeline

```
Raw Data
↓
Download + Preprocessing
↓
Embedding Generation (Text / Image / CF / VLM)
↓
Multimodal Fusion (optional)
↓
Quantization (RQ-VAE / OPQ / PQ / RKMeans)
↓
Generative Recommender (TIGER / RPG / LETTER / LLMs)
↓
Inference (Beam Search / Prefix-tree / Contrastive Rerank)
```


---


# 🚀 Quick Start


**Requirements**
- Python **3.10** (recommended)
- CUDA 11.8+ (for GPU acceleration)
- PyTorch, CUDA, and other dependencies will be installed automatically via `requirements.txt`

```bash
pip install -r requirements.txt
```
## 1 Data Preprocessing

We provide a dedicated submodule for downloading, cleaning, and extracting embeddings (Text/Image/CF).

👉 **See detailed tutorial:**  
[GenRec-Factory Data Processing & Embedding Guide](./preprocessing/ReadMe.md)


## 2 Quantization

Convert dense embeddings into discrete Semantic IDs (SIDs)

```bash
cd quantization

python main.py \
  --model_name rqvae \
  --dataset_name amazon-musical-instruments-23 \
  --embedding_modality text \
  --embedding_model sentence-t5-base
```

## 3 Generative Recommendation Models

Train a generative recommender using the generated SIDs.

```bash
cd recommendation

python main.py \
  --model ReGen \
  --dataset amazon-musical-instruments-23 \
  --quant_method rqvae
```
