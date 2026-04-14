import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Callable
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

SFT_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
)


def build_semantic_special_tokens(vocab_sizes: List[int]) -> List[str]:
    tokens = []
    for level, vocab_size in enumerate(vocab_sizes):
        for value in range(vocab_size):
            tokens.append(f"<SID_{level}_{value}>")
    return tokens

class BaseTokenizer:
    """Tokenizer 基類 (保持不變)"""
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        self.config = config
        self.item_to_code_map = item_to_code_map
        self.pad_token_id = config['token_params']['pad_token_id']
        self.mask_token_id = config['token_params'].get('mask_token_id')
        self.cls_token_id = config['token_params'].get('cls_token_id')
        self.sep_token_id = config['token_params'].get('sep_token_id')
        self.code_len = config['code_len']
        self.item_pad_id = 0 
        self.code_pad_list = [self.pad_token_id] * self.code_len
        self.max_len = config['model_params']['max_len']

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _pad_sequences(self, sequences: List[torch.Tensor], padding_side: str, padding_value: int) -> torch.Tensor:
        """
        輔助函數：執行 Padding (使用更可靠的 left-padding)。
        """
        # sequences: list of tensors, where each tensor is a flattened sequence of valid codes
        
        # 獲取每個序列的實際長度
        lengths = [len(s) for s in sequences]
        max_len = max(lengths) if lengths else 0

        if padding_side == 'right':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                # torch.nn.functional.pad only supports same padding on both sides easily
                # Manual padding is clearer here
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((s, padding)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch

        elif padding_side == 'left':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((padding, s)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch
        else:
            raise ValueError(f"不支持的 padding_side: {padding_side}")

class AdaDiffTokenizer(BaseTokenizer):
    """
    為 AdaDiff 構造輸入，強制左填充並支持訓練/評估兩種掩碼策略。
    """

    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]], is_training: bool):
        super().__init__(config, item_to_code_map)
        self.is_training = is_training
        self.history_mask_prob = config["model_params"].get("history_mask_prob", 0.15)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sequences = []
        seq_lens = []
        hist_token_lens = []
        target_codes_list = []

        for item in batch:
            hist_ids_0based = item["history"]
            tgt_id_0based = item["target"]

            valid_hist_ids = [x for x in hist_ids_0based if x != self.item_pad_id]
            valid_hist_codes = [self.item_to_code_map.get(x + 1, self.code_pad_list) for x in valid_hist_ids]
            hist_tokens = [code for codes in valid_hist_codes for code in codes]
            target_codes = self.item_to_code_map.get(tgt_id_0based + 1, self.code_pad_list)

            seq = [self.cls_token_id] + hist_tokens + [self.sep_token_id] + target_codes
            sequences.append(torch.tensor(seq, dtype=torch.long))
            seq_lens.append(len(seq))
            hist_token_lens.append(len(hist_tokens))
            target_codes_list.append(target_codes)

        padded = self._pad_sequences(
            sequences, padding_side="left", padding_value=self.pad_token_id
        )
        attention_mask = (padded != self.pad_token_id).long()

        labels = torch.full_like(padded, -100)
        max_len = padded.shape[1] if padded.numel() > 0 else 0
        target_start = max_len - self.code_len

        for i, seq_len in enumerate(seq_lens):
            pad_len = max_len - seq_len
            # History masking (training only)
            if self.is_training and hist_token_lens[i] > 0:
                hist_start = pad_len + 1  # skip [CLS]
                hist_end = hist_start + hist_token_lens[i]
                for idx in range(hist_start, hist_end):
                    if random.random() < self.history_mask_prob:
                        padded[i, idx] = self.mask_token_id
                        # labels 保持 -100

            # Target masking
            target_slice = slice(target_start, max_len)
            if self.is_training:
                # 訓練階段：隨機遮蔽 0%~100% 的 target token，保留部分上下文以穩定收斂
                mask_ratio = np.random.uniform(0.0, 1.0)
                num_to_mask = max(1, int(np.ceil(mask_ratio * self.code_len)))
                mask_positions = set(np.random.choice(self.code_len, size=num_to_mask, replace=False).tolist())
                for offset in range(self.code_len):
                    abs_idx = target_start + offset
                    if offset in mask_positions:
                        labels[i, abs_idx] = padded[i, abs_idx]
                        padded[i, abs_idx] = self.mask_token_id
                    else:
                        labels[i, abs_idx] = -100
            else:
                # 評估：target 全掩碼，labels 僅保存真值供評估使用
                labels[i, target_slice] = -100
                padded[i, target_slice] = self.mask_token_id

        target_codes_tensor = torch.tensor(target_codes_list, dtype=torch.long)

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_codes": target_codes_tensor,
        }

class GenerativeTokenizer(BaseTokenizer):
    """
    為 TIGER 和 GPT-2 準備數據 (最終修正版 v3 - 移除 Padding)。
    核心：
    1. 移除 Item Padding 0。
    2. 壓平有效 Code。
    3. 使用可靠的 Padding 函數填充。
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]], padding_side: str):
        super().__init__(config, item_to_code_map)
        if padding_side not in ('left', 'right'):
            raise ValueError("padding_side 必須是 'left' 或 'right'")
        self.padding_side = padding_side
        logger.info(f"GenerativeTokenizer 初始化, padding_side='{padding_side}'")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        batch_sequences_flat_valid = [] # 存儲每個樣本壓平後的 *有效* code token
        target_codes = []

        logger.debug(f"--- Tokenizer Start Batch (Size: {len(batch)}) ---") 

        for i, item in enumerate(batch):
            hist_ids_0based = item['history'] # e.g., [0, 0, 5, 10] (left-padded from dataset)
            tgt_id_0based = item['target']
            
            logger.debug(f"Sample {i} Raw History Item IDs: {hist_ids_0based}") 

            # 1. ✅ 過濾掉 Item Padding (0)
            valid_hist_ids = [x for x in hist_ids_0based if x != self.item_pad_id]
            logger.debug(f"Sample {i} Valid History Item IDs: {valid_hist_ids}") 

            # 2. 將有效 Item IDs 轉換為 Code 列表
            valid_hist_codes = []
            if valid_hist_ids: 
                valid_hist_codes = [self.item_to_code_map.get(x + 1, self.code_pad_list) 
                                    for x in valid_hist_ids]

            # 3. 壓平 *有效* Code Token 序列
            seq_flat_valid = [code for item_codes in valid_hist_codes for code in item_codes]
            logger.debug(f"Sample {i} Flattened Valid Codes (len={len(seq_flat_valid)}): {seq_flat_valid[:20]}...{seq_flat_valid[-20:]}") 
            
            batch_sequences_flat_valid.append(torch.tensor(seq_flat_valid, dtype=torch.long))

            # Target (保持不變)
            t_code = self.item_to_code_map.get(tgt_id_0based + 1, self.code_pad_list)
            target_codes.append(t_code)

        # 4. ✅ 對壓平後的 *有效* 序列進行 Padding (使用修正後的函數)
        padded_histories = self._pad_sequences(
            batch_sequences_flat_valid, 
            padding_side=self.padding_side, 
            padding_value=self.pad_token_id 
        )
        logger.debug(f"Padded Histories Shape: {padded_histories.shape}") 
        if len(batch) > 0 and padded_histories.numel() > 0: # Add check for empty tensor
            logger.debug(f"Padded Histories[0] (first 30): {padded_histories[0][:30].tolist()}") 
            logger.debug(f"Padded Histories[0] (last 30): {padded_histories[0][-30:].tolist()}") 
        
        # 5. 生成 Attention Mask
        attention_masks = (padded_histories != self.pad_token_id).long() 
        if len(batch) > 0 and attention_masks.numel() > 0: # Add check for empty tensor
            logger.debug(f"Attention Mask[0] (first 30): {attention_masks[0][:30].tolist()}") 
            logger.debug(f"Attention Mask[0] (last 30): {attention_masks[0][-30:].tolist()}") 
        
        # 6. Target Tensor (保持不變)
        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        
        logger.debug(f"--- Tokenizer End Batch ---") 

        # 預期輸出:
        # GPT-2 (left): input_ids=[0,...,0, C6.., C11..] mask=[0,...,0, 1.., 1..]
        # TIGER (right): input_ids=[C6.., C11.., 0,...,0] mask=[1.., 1.., 0,...,0]
        
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
        }

# --- RetrievalTokenizer 和 get_tokenizer 保持不變 ---
# --- RetrievalTokenizer (✅ 已补全) ---
class RetrievalTokenizer(BaseTokenizer):
    """
    为 RPG 模型准备数据。
    GenRecDataset 返回: {'history': [0, 0, 5, 10], 'target': 20} (0-based IDs)
    RPG (forward) 期望:
        - 'input_ids': [0, 0, 6, 11] (1-based IDs, 0-padded)
        - 'attention_mask': [0, 0, 1, 1]
        - 'labels_seq': [-100, -100, -100, 20] (0-based target, -100-padded)
    RPG (evaluate_step) 期望:
        - 'input_ids': (同上)
        - 'attention_mask': (同上)
        - 'target_ids': [20] (0-based target)
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        super().__init__(config, item_to_code_map)
        logger.info(f"RetrievalTokenizer (RPG) 初始化, max_len={self.max_len}")
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        # 1. 初始化列表
        history_seqs_1based = []
        label_seqs = []
        target_ids_0based = []

        for item in batch:
            hist_0based = item['history'] # e.g., [0, 0, 5, 10]
            target_0based = item['target']  # e.g., 20

            # 2. 转换 History (Input IDs)
            # 过滤掉 padding (0)，然后将 0-based IDs 转换为 1-based IDs
            valid_hist_0based = [x for x in hist_0based if x != self.item_pad_id]
            valid_hist_1based = [x + 1 for x in valid_hist_0based]
            
            # 截断到 max_len
            valid_hist_1based = valid_hist_1based[-self.max_len:]
            seq_len = len(valid_hist_1based)
            history_seqs_1based.append(torch.tensor(valid_hist_1based, dtype=torch.long))

            # 3. 准备 labels_seq (用于训练)
            # 只有最后一个时间步有-label，其他都是 -100
            labels = [-100] * seq_len
            if seq_len > 0:
                labels[-1] = target_0based # 在最后一个有效位置放 0-based 目标
            label_seqs.append(torch.tensor(labels, dtype=torch.long))

            # 4. 准备 target_ids (用于评估)
            target_ids_0based.append(target_0based)

        # 5. Pad Input IDs (使用 0 进行 right-padding)
        padded_input_ids = pad_sequence(
            history_seqs_1based, 
            batch_first=True, 
            padding_value=self.item_pad_id # self.item_pad_id 必须是 0
        )
        
        # 6. 创建 Attention Mask
        attention_mask = (padded_input_ids != self.item_pad_id).long()
        
        # 7. Pad Labels Seq (使用 -100 进行 right-padding)
        padded_labels_seq = pad_sequence(
            label_seqs, 
            batch_first=True, 
            padding_value=-100 # RPG (forward) 期望 -100
        )
        
        # 8. Stack Target IDs
        target_ids = torch.tensor(target_ids_0based, dtype=torch.long)

        # 确保 padding 后的长度一致
        # (pad_sequence 会自动处理)
        
        return {
            'input_ids': padded_input_ids,    # (B, L_item) 1-based IDs, 0-padded
            'attention_mask': attention_mask, # (B, L_item)
            'labels_seq': padded_labels_seq,  # (B, L_item) 0-based target, -100-padded
            'target_ids': target_ids          # (B,) 0-based target
        }


class LCRecTokenizer(BaseTokenizer):
    """
    Build Alpaca/SFT-style instruction data with Semantic ID special tokens.
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]], is_training: bool):
        super().__init__(config, item_to_code_map)
        self.is_training = is_training
        model_params = config["model_params"]
        model_name_or_path = model_params["model_name_or_path"]
        from transformers import AutoTokenizer

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=model_params.get("trust_remote_code", False),
        )
        self.semantic_special_tokens = build_semantic_special_tokens(config["vocab_sizes"])
        self.llm_tokenizer.add_special_tokens(
            {"additional_special_tokens": self.semantic_special_tokens}
        )
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.max_text_len = model_params.get("max_text_len", 512)
        self.history_sep = model_params.get("history_sep", ", ")
        self.add_history_prefix = model_params.get("history_add_index_prefix", False)
        self.instruction_template = model_params.get(
            "instruction_template",
            "Given the user's historical interactions in chronological order: {history}\n"
            "Predict the next item Semantic ID.",
        )

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        return SFT_PROMPT.format(instruction=instruction, input_text=input_text)

    def _build_training_instance(
        self,
        instruction: str,
        input_text: str,
        response: str,
    ) -> Dict[str, torch.Tensor]:
        prompt_text = self._format_prompt(instruction, input_text)
        prompt_ids = self.llm_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.llm_tokenizer(f" {response}", add_special_tokens=False)["input_ids"]
        eos_id = self.llm_tokenizer.eos_token_id

        input_ids = prompt_ids + response_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + response_ids + [eos_id]

        return {
            "input_ids": torch.tensor(input_ids[: self.max_text_len], dtype=torch.long),
            "labels": torch.tensor(labels[: self.max_text_len], dtype=torch.long),
        }

    def _build_eval_instance(
        self,
        instruction: str,
        input_text: str,
        target_text: str,
        target_item_id: int,
    ) -> Dict[str, torch.Tensor]:
        prompt_text = self._format_prompt(instruction, input_text)
        prompt_ids = self.llm_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.llm_tokenizer(target_text, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": torch.tensor(prompt_ids[: self.max_text_len], dtype=torch.long),
            "target_token_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_item_id": torch.tensor(target_item_id, dtype=torch.long),
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.is_training:
            features = []
            for item in batch:
                features.append(
                    self._build_training_instance(
                        instruction=item["instruction"],
                        input_text=item.get("input", ""),
                        response=item["output"],
                    )
                )

            input_ids = pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=self.llm_tokenizer.pad_token_id,
            )
            labels = pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=-100,
            )
            attention_mask = (input_ids != self.llm_tokenizer.pad_token_id).long()
            labels = labels.masked_fill(input_ids == self.llm_tokenizer.pad_token_id, -100)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        eval_features = []
        for item in batch:
            eval_features.append(
                self._build_eval_instance(
                    instruction=item["instruction"],
                    input_text=item.get("input", ""),
                    target_text=item["output"],
                    target_item_id=int(item["target_item"]),
                )
            )

        input_ids = pad_sequence(
            [feature["input_ids"] for feature in eval_features],
            batch_first=True,
            padding_value=self.llm_tokenizer.pad_token_id,
        )
        attention_mask = (input_ids != self.llm_tokenizer.pad_token_id).long()
        target_token_ids = torch.stack(
            [feature["target_token_ids"] for feature in eval_features],
            dim=0,
        )
        target_item_ids = torch.stack(
            [feature["target_item_id"] for feature in eval_features],
            dim=0,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_token_ids": target_token_ids,
            "target_item_ids": target_item_ids,
        }

def get_tokenizer(model_name: str, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]) -> Callable:
    """
    (保持之前的版本不變)
    """
    model_name = model_name.upper()
    if model_name == 'ADADIFF':
        return {
            'train': AdaDiffTokenizer(config, item_to_code_map, is_training=True),
            'eval': AdaDiffTokenizer(config, item_to_code_map, is_training=False)
        }
    if model_name == 'LCREC':
        return {
            'train': LCRecTokenizer(config, item_to_code_map, is_training=True),
            'eval': LCRecTokenizer(config, item_to_code_map, is_training=False)
        }
    if model_name == 'TIGER':
        return GenerativeTokenizer(config, item_to_code_map, padding_side='right')
    elif 'GPT2' in model_name or 'LLM' in model_name:
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')
    elif model_name == 'RPG':
        return RetrievalTokenizer(config, item_to_code_map)
    else:
        logger.warning(f"未知模型: {model_name}，使用 left-padding GenerativeTokenizer")
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')
