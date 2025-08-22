import os
import argparse
import json
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
# 确保您已经安装了 OpenAI 的 clip 包: pip install git+https://github.com/openai/CLIP.git
from clip import clip

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_id2item_dict(item2id_file):
    with open(item2id_file, 'r') as fp:
        all_item2id = fp.readlines()
        
    id2item = {}
    for line in all_item2id:
        try:
            item, item_id = line.strip().split('\t')
            id2item[item_id] = item
        except ValueError:
            continue # 跳过格式不正确的行
    return id2item

def get_feature(args):
    print(f"开始处理: Dataset={args.dataset}, Version={args.data_version}, Backbone={args.backbone}")

    # 构建图片和图片信息文件的根目录
    # 例如: /userhome/dataset/amazon14/Images
    image_base_path = os.path.join(args.image_root)
    
    # 存放具体数据集图片的目录
    # 例如: /userhome/dataset/amazon14/Images/Baby
    image_file_path = os.path.join(image_base_path, args.dataset)
    
    # 图片信息 .json 文件的完整路径
    # 例如: /userhome/dataset/amazon14/Images/Baby_images_info.json
    images_info_path = os.path.join(image_base_path, f'{args.dataset}_images_info.json')

    # `save_root` 指向的是上一步预处理脚本的输出目录
    # 例如: /userhome/dataset/MQL4GRec/Baby
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f'{args.dataset}.item2id')

    # --- 2. 加载所需文件 ---
    print(f"从 {item2id_file} 加载 item2id 映射...")
    id2item = get_id2item_dict(item2id_file)
    if not id2item:
        print(f"错误: 未能从 {item2id_file} 加载任何物品ID。请检查该文件是否正确生成。")
        return

    print(f"从 {images_info_path} 加载图片信息...")
    images_info = load_json(images_info_path)
    
    # --- 3. 加载 CLIP 模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'使用设备: {device}')
    print('加载 CLIP 模型...')
    model, preprocess = clip.load(args.backbone, device=device, download_root=args.model_cache_dir)

    # --- 4. 提取特征 ---
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(id2item)), desc="提取图像特征"):
            item = id2item.get(str(i))
            if not item:
                continue

            # 获取图片文件名列表，并确保它是一个列表
            image_name_list = images_info.get(item, [])
            if not image_name_list:
                # 如果没有图片，用一个零向量作为占位符
                embeddings.append(torch.zeros(model.visual.output_dim))
                continue
            
            image_name = image_name_list[0]
            image_file = os.path.join(image_file_path, image_name)

            try:
                image = Image.open(image_file).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_tensor)
                image_feature = image_features[0].cpu()
            except Exception as e:
                print(f"\n处理图片时发生错误: {e}")
                print(f"文件路径: {image_file}")
                # 出现错误时，同样使用零向量占位
                image_feature = torch.zeros(model.visual.output_dim)

            embeddings.append(image_feature)
            
    # --- 5. 保存特征文件 ---
    embeddings = torch.stack(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    backbone_name = args.backbone.replace('/', '-')
    
    # 将特征文件保存在预处理数据的输出目录中
    save_file = os.path.join(processed_data_path, f'{args.dataset}.emb-{backbone_name}.npy')
    
    print(f"正在保存特征到: {save_file}")
    np.save(save_file, embeddings)
    print("特征提取并保存成功！")

def parse_args():
    parser = argparse.ArgumentParser(description="使用CLIP模型为Amazon数据集提取图像特征。")
    # --- MODIFIED ---
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'], help='数据集年份版本 (14 或 18)')
    parser.add_argument('--dataset', type=str, default='Home', help='例如 Baby, All_Beauty, Home 等')
    # 将 image_root 的默认值改为更通用的根目录
    parser.add_argument('--image_root', type=str, default="../datasets/amazon14/Images", help='包含 amazon14/ 和 amazon18/ 的根目录')
    # `save_root` 是上一步(2_process.sh)的输出目录，也是本脚本的输入和输出目录
    parser.add_argument('--save_root', type=str, default="../datasets", help='预处理后数据的根目录')
    parser.add_argument('--backbone', type=str, default='ViT-L/14', help='CLIP模型的主干网络')
    parser.add_argument('--model_cache_dir', type=str, default='../cache_models/clip', help='CLIP模型下载缓存目录')
    # gpu_id 参数通常由 CUDA_VISIBLE_DEVICES 控制，这里可以移除或保留
    # parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU') 
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    # 如果您想用 gpu_id 参数控制GPU，可以取消下面这行的注释
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    get_feature(args)
    
