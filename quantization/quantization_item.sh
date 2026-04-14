python main.py --model_name rqvae --dataset_name ml-100k --embedding_modality text  --embedding_model sentence-t5-base

# 多模态
python main.py \
  --model_name mm_rqvae \
  --dataset_name Baby \
  --text_embedding_model text-embedding-3-large \
  --image_embedding_model clip-vit-base-patch32 