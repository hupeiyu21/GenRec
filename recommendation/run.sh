export CUDA_VISIBLE_DEVICES=6
python main.py --model TIGER --dataset Musical_Instruments --quant_method rkmeans --embedding_modality cf

export CUDA_VISIBLE_DEVICES=6
python main.py --dataset ml-100k --quant_method rqvae  --model TIGER
export CUDA_VISIBLE_DEVICES=7
python main.py --model RPG --dataset Beauty --quant_method opq