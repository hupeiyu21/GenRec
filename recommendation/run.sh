export CUDA_VISIBLE_DEVICES=6
python main.py --model TIGER --dataset Musical_Instruments --quant_method rkmeans --embedding_modality cf

export CUDA_VISIBLE_DEVICES=6
python main.py --dataset amazon-musical-instruments-23 --quant_method opq  --model RPG

python main.py --dataset amazon-video-games-23 --model TIGER

python main.py --dataset amazon-industrial-scientific-23

python main.py --dataset Musical_Instruments
export CUDA_VISIBLE_DEVICES=7
python main.py --model RPG --dataset Beauty --quant_method opq