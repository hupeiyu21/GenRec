import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
import ast  # 用于 Amazon 2014
# 假设 utils.py 在同一目录或 Python 路径中
from utils import check_path, clean_text, write_json_file, write_remap_index, load_json

# --- Amazon 2018/2014 特定的元数据加载逻辑 ---
def load_meta_items_amazon(file_path, data_version):
    """
    加载亚马逊元数据文件，智能兼容 2014 和 2018 年的数据集格式。
    无论输入是哪个版本，都输出统一格式的字典。
    """
    items = {}
    parser = ast.literal_eval if data_version == '14' else json.loads

    with gzip.open(file_path, "rt", encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Load Amazon metas"):
            try:
                data = parser(line)
                item_id = data.get("asin")
                if not item_id:
                    continue

                # 定义统一的数据结构
                unified_info = {
                    "title": "", 
                    "description": "", 
                    "brand": "", 
                    "categories": ""
                }

                # --- 兼容性逻辑 ---
                if data_version == '18':
                    unified_info['title'] = clean_text(data.get('title', ''))
                    unified_info['description'] = clean_text(' '.join(data.get('description', [])))
                    unified_info['brand'] = data.get('brand', '').replace("by\n", "").strip()
                    # 2018版的'category'是列表
                    cats = [cat.strip() for cat in data.get('category', []) if "</span>" not in cat]
                    unified_info['categories'] = ",".join(cats).strip()
                
                elif data_version == '14':
                    unified_info['title'] = clean_text(data.get('title', ''))
                    unified_info['description'] = clean_text(data.get('description', ''))
                    unified_info['brand'] = data.get('brand', '').replace("by\n", "").strip()
                    # 2014版的'categories'是列表的列表
                    cats_list = data.get('categories', [[]])
                    if cats_list and isinstance(cats_list, list) and len(cats_list) > 0:
                        cats = [cat.strip() for cat in cats_list[0] if "</span>" not in cat]
                        unified_info['categories'] = ",".join(cats).strip()
                
                items[item_id] = unified_info
            
            except (ValueError, SyntaxError, TypeError, KeyError):
                continue
    return items

import csv
from tqdm import tqdm

def load_item_meta_recbole(file_path, sep='\t'):
    """
    加载 RecBole 风格的 item 元数据文件，动态解析列名，
    并将列名中 ':' 之后的类型信息去除。

    返回:
        items: dict
            {
              item_id: {
                  field_name: value,
                  ...
              }
            }
    """
    items = {}

    with open(file_path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=sep)

        # 读取表头
        raw_header = next(reader)
        header = [h.split(':')[0] for h in raw_header]

        # 找到 item_id 列索引
        if 'item_id' not in header:
            raise ValueError("item metadata must contain 'item_id' column")

        item_id_idx = header.index('item_id')

        for row in tqdm(reader, desc="Load RecBole item meta"):
            if not row or len(row) != len(header):
                continue

            item_id = row[item_id_idx]
            if not item_id:
                continue

            meta = {}
            for i, field in enumerate(header):
                if i == item_id_idx:
                    continue

                value = row[i]
                if value == '' or value == 'None':
                    continue

                meta[field] = value

            items[item_id] = meta

    return items


def preprocess_recbole(args):
    """
    处理 recbole 数据集。
    """
    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # 动态构造输入文件根目录
    input_root_path = os.path.join(args.input_path, args.dataset)
    # 动态构造 ratings 文件路径
    rating_file_path = os.path.join(input_root_path,f'{args.dataset}.inter')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")

    users, items, rating_inters = set(), set(), set()

    with open(rating_file_path, 'r', encoding='utf-8') as fp:
        next(fp)  # 跳过第一行（表头）
        for line in tqdm(fp, desc='Load ratings'):
            try:
                user, item, rating, time = line.strip().split('\t')
                users.add(user)
                items.add(item)
                rating_inters.add((user, item, float(rating), int(float(time))))
            except ValueError:
                continue

    # 动态构造 meta 文件路径
    meta_file_path = os.path.join(input_root_path, f'{args.dataset}.item')
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_file_path}")

    # 调用 RecBole 专属的 load_item_meta_recbole
    meta_items = load_item_meta_recbole(meta_file_path)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)
    
    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by meta items"):
        if inter[1] in meta_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after meta filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)
    
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')
    return rating_inters, meta_items

def preprocess_amazon(args):
    """
    处理 Amazon '14 或 '18 数据集。
    """
    print('Process Amazon rating data: ')
    print(' Dataset: ', args.dataset)
    print(' Data Version: ', args.data_version)

    # 动态构造输入文件根目录
    input_root_path = os.path.join(args.input_path, f'amazon{args.data_version}')

    # (已移除 Amazon 脚本中未使用的 images_info 加载)

    # 动态构造 ratings 文件路径
    rating_file_path = os.path.join(input_root_path, 'Ratings', f'{args.dataset}.csv')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")
    
    # 调用通用的 load_ratings
    _, _, rating_inters = load_ratings(rating_file_path)

    # 动态构造 meta 文件路径
    meta_file_path = os.path.join(input_root_path, 'Metadata', f'meta_{args.dataset}.json.gz')
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_file_path}")

    # 调用 Amazon 专属的 load_meta_items_amazon
    meta_items = load_meta_items_amazon(meta_file_path, args.data_version)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)
    
    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by meta items"):
        if inter[1] in meta_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after meta filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)
    
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')
    return rating_inters, meta_items

# --- MovieLens 特定的元数据加载逻辑 ---
def load_meta_items_movielens(file_path):
    """加载电影元数据"""
    movies = {}
    with open(file_path, 'r', encoding='utf-8') as fp:
        movies_data = json.load(fp)
        
    for movie_id, data in movies_data.items():
        movies[movie_id] = {
            'title': clean_text(data.get('title', '')),
            'description': clean_text(data.get('description', '')),
            'genres': data.get('genres', []),
            'year': data.get('year', None)
        }
    return movies

def preprocess_movielens(args):
    """处理 MovieLens 数据集"""
    print('Process MovieLens rating data: ')
    print(' Dataset: ', args.dataset)

    # 构建输入文件路径
    input_root_path = os.path.join(args.input_path, args.dataset, 'processed')
    
    # 加载评分数据
    rating_file_path = os.path.join(input_root_path, f'{args.dataset}.csv')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")
    
    # 调用通用的 load_ratings
    _, _, rating_inters = load_ratings(rating_file_path)

    # 加载电影元数据
    movies_file_path = os.path.join(input_root_path, f'{args.dataset}.item.json')
    if not os.path.exists(movies_file_path):
        raise FileNotFoundError(f"Movies file not found: {movies_file_path}")
    
    # 调用 MovieLens 专属的 load_meta_items_movielens
    movie_items = load_meta_items_movielens(movies_file_path)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)
    
    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by movie metadata"):
        if inter[1] in movie_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after metadata filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)
    
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')
    return rating_inters, movie_items

# =================================================================
# ============ 以下是两个脚本完全共享的通用函数 ============
# =================================================================

def load_ratings(file):
    """
    (通用) 加载 .csv 格式的评分数据
    格式: item, user, rating, time
    """
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                continue
    return users, items, inters

def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count

def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count

def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)

def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    """(通用) K-core 过滤器"""
    new_inters = []
    # 注意：can_items 逻辑在特定于数据集的预处理函数中执行了
    if can_items:
        print('\nFiltering by meta items (Deprecated in unified script): ')
        for unit in tqdm(inters):
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))
        
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)
        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters

def make_inters_in_order(inters):
    """(通用) 按用户和时间戳排序交互"""
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in tqdm(inters):
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in tqdm(user2inters):
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item:
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters

def convert_inters2dict(inters):
    """
    (通用) 将原始交互映射为 user2items, user2index, item2index。
    """
    all_users = {u for (u, i, r, t) in inters}
    all_items = {i for (u, i, r, t) in inters}

    users_sorted = sorted(all_users)
    items_sorted = sorted(all_items)
    user2index = {u: idx for idx, u in enumerate(users_sorted)}
    item2index = {i: idx for idx, i in enumerate(items_sorted)}

    user2items = collections.defaultdict(list)
    for u, it, r, ts in inters:
        uid = user2index[u]
        iid = item2index[it]
        user2items[uid].append(iid)

    return user2items, user2index, item2index

def generate_data(args, rating_inters):
    """(通用) 划分训练/验证/测试集"""
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # 确保用户至少有3次交互 (k-core=5 应该保证了这一点)
        if len(inters) < 3:
            continue
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + len(valid_inters[u_index]) + len(test_inters[u_index])
    
    # 过滤掉那些在 K-core 之后但在划分时少于3个交互的用户
    valid_uids = set(train_inters.keys())
    user2items = {uid: items for uid, items in user2items.items() if uid in valid_uids}
    
    print(f"Total users after split (>=3 items): {len(user2items)}")
    return user2items, train_inters, valid_inters, test_inters, user2index, item2index

def convert_to_atomic_files(args, train_data, valid_data, test_data, max_history_len=50, use_sliding_window=True):
    """
    (通用) 保存为 JSONL 文件
    """
    print('Convert dataset to JSONL:')
    print(' Dataset: ', args.dataset)

    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    output_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # --- Train ---
    train_path = os.path.join(output_dir, f"{args.dataset}.train.jsonl")
    with open(train_path, 'w') as f:
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            if seq_len < 1: continue # 至少需要1个历史 + 1个目标

            if use_sliding_window:
                # 滑动窗口：逐步生成 prefix → target
                for target_idx in range(1, seq_len):
                    target_item = item_seq[target_idx] # 目标是第 target_idx 个
                    seq = item_seq[:target_idx] # 历史是 0 到 target_idx-1
                    seq = seq[-max_history_len:] # 截断
                    json.dump({"user": str(uid), "history": seq, "target": target_item}, f)
                    f.write("\n")
            else:
                # 只取最后一个
                if seq_len > 0:
                    target_item = item_seq[-1]
                    seq = item_seq[:-1][-max_history_len:]
                    json.dump({"user": str(uid), "history": seq, "target": target_item}, f)
                    f.write("\n")

    # --- Valid ---
    valid_path = os.path.join(output_dir, f"{args.dataset}.valid.jsonl")
    with open(valid_path, 'w') as f:
        for uid in uid_list:
            item_seq = train_data[uid][-max_history_len:]
            target_item = valid_data[uid][0]
            json.dump({"user": str(uid), "history": item_seq, "target": target_item}, f)
            f.write("\n")

    # --- Test ---
    test_path = os.path.join(output_dir, f"{args.dataset}.test.jsonl")
    with open(test_path, 'w') as f:
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-max_history_len:]
            target_item = test_data[uid][0]
            json.dump({"user": str(uid), "history": item_seq, "target": target_item}, f)
            f.write("\n")

    print(f"JSONL files saved to {output_dir}")

def parse_args():
    """(通用) 统一的参数解析器"""
    parser = argparse.ArgumentParser()
    
    # 新增：用于分发任务的参数
    parser.add_argument('--dataset_type', type=str, required=True, choices=['amazon', 'movielens','recbole'],
                        help='Type of the dataset to process (amazon or movielens)')
    
    # 通用参数
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (e.g., Home, Baby, ml-1m, ml-20m)')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../datasets', 
                        help='Root path containing dataset folders (e.g., amazon14/, amazon18/, ml-1m/)')
    parser.add_argument('--output_path', type=str, default='../datasets',
                        help='Root path to save processed data')
    
    # Amazon 专用参数
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'],
                        help='Amazon dataset version (14 or 18). Only used if dataset_type=amazon.')
    
    return parser.parse_args()

# =================================================================
# =================== 主程序入口 (Main) ===================
# =================================================================

if __name__ == '__main__':
    args = parse_args()
    
    print('\n' + '=' * 20)
    print(f"Start processing dataset: {args.dataset} (Type: {args.dataset_type})")
    print('=' * 20 + '\n')
        
    # --- 1. 调度特定于数据集的预处理 ---
    if args.dataset_type == 'amazon':
        rating_inters, meta_items = preprocess_amazon(args)
    elif args.dataset_type == 'movielens':
        rating_inters, meta_items = preprocess_movielens(args)
    elif args.dataset_type == 'recbole':
        rating_inters, meta_items = preprocess_recbole(args)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    
    # --- 2. 执行通用的数据划分 ---
    # 无论上面走了哪个分支，这里都拿到了 rating_inters 和 meta_items
    all_inters, train_inters, valid_inters, test_inters, user2index, item2index = generate_data(args, rating_inters)
    
    # --- 3. 执行通用的文件保存 ---
    output_dataset_path = os.path.join(args.output_path, args.dataset)
    check_path(output_dataset_path)

    # 保存 .inter.json
    write_json_file(all_inters, os.path.join(output_dataset_path, f'{args.dataset}.inter.json'))
    
    # 保存 .train.jsonl, .valid.jsonl, .test.jsonl
    # (注意：我修改了 convert_to_atomic_files 中的滑动窗口逻辑，使其更符合标准)
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    # 准备并保存 .item.json
    item2feature = collections.defaultdict(dict)
    for item_str, item_id_int in item2index.items():
        # 确保来自交互的 item 存在于元数据中
        if item_str in meta_items:
            item2feature[item_id_int] = meta_items[item_str]

    print("Total users:", len(user2index))
    print("Total items (with meta):", len(item2feature))
    print("Total items (in inters):", len(item2index))
    print("Total interactions:", len(rating_inters))
    sparsity = 1 - len(rating_inters) / (len(user2index) * len(item2index))
    print(f"Sparsity: {sparsity * 100:.4f}%")

    write_json_file(item2feature, os.path.join(output_dataset_path, f'{args.dataset}.item.json'))
    
    # 保存映射文件
    write_remap_index(user2index, os.path.join(output_dataset_path, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(output_dataset_path, f'{args.dataset}.item2id'))
    
    print(f"\nFinished processing dataset: {args.dataset}")