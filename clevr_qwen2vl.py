import os
import ast
import random
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

from transformers import (
    AutoProcessor,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from collections import Counter

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"


class CLEVRXTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        processor,
        max_explanation_len: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.processor = processor
        self.max_explanation_len = max_explanation_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["file"])
        image = Image.open(img_path).convert("RGB")

        question = row["question"]
        answer = str(row["answer"]).strip()

        # explanation 是字符串形式的 Python list
        exp_list = ast.literal_eval(row["explanation"])
        explanation = random.choice(exp_list).strip()

        if len(explanation) > self.max_explanation_len:
            explanation = explanation[: self.max_explanation_len]

        user_text = (
            "You are a visual reasoning assistant for synthetic CLEVR-style scenes.\n"
            "Carefully look at the image and answer the question.\n\n"
            f"Question: {question}\n\n"
            "First give a short answer, then a detailed reasoning explanation.\n"
            "Answer MUST be a single word or integer token from the dataset vocabulary.\n"
            "Do NOT add units or extra words in the Answer line.\n"
            "Format:\n"
            "Answer: <short answer>\n"
            "Explanation: <step-by-step reasoning>"
        )

        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]

        assistant_text = f"Answer: {answer}\nExplanation: {explanation}"

        # 只含 user 的 prompt（用于 mask 掉 loss）
        prompt_messages = [
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # user + assistant 的完整文本（训练目标）
        full_messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ]
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "image": image,
            "full_text": full_text,
            "prompt_text": prompt_text,
        }


@dataclass
class QwenVLDataCollator:
    processor: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        full_texts = [item["full_text"] for item in batch]
        prompt_texts = [item["prompt_text"] for item in batch]

        # Encode full (user + assistant) text for training
        enc_full = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Encode only the user prompt (for masking loss). No gradients needed here.
        with torch.no_grad():
            enc_prompt = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )

        input_ids = enc_full["input_ids"]
        pad_token_id = self.processor.tokenizer.pad_token_id

        labels = input_ids.clone()
        # First ignore all padding tokens
        labels[labels == pad_token_id] = -100

        prompt_ids = enc_prompt["input_ids"]

        # For each sample, mask out the user+image tokens so that the loss
        # is only computed on assistant outputs.
        for i in range(input_ids.size(0)):
            # number of non-pad tokens in the prompt sequence
            prompt_len = (prompt_ids[i] != pad_token_id).sum()
            if prompt_len > 0:
                labels[i, :prompt_len] = -100

        enc_full["labels"] = labels
        return enc_full


def parse_answer_explanation(text: str):
    import re

    def normalize_answer(ans: str) -> str:
        """
        Clean and normalize the short answer extracted from the model output.
        We aggressively strip punctuation and whitespace, and map common variants
        (e.g., spelled-out numbers, grey/gray) to a canonical single token.
        """
        ans = ans.strip().lower()
        # keep only the first line
        ans = ans.splitlines()[0].strip()
        # cut off anything after an "explanation:" that accidentally got included
        ans = re.split(r"\bexplanation\s*:", ans, flags=re.IGNORECASE)[0]
        # remove common punctuation and brackets
        ans = re.sub(r"[.,!?]", " ", ans)
        ans = ans.replace("(", " ").replace(")", " ")
        # collapse multiple spaces
        ans = re.sub(r"\s+", " ", ans).strip()
        if not ans:
            return ""

        tokens = ans.split()
        head = tokens[0]

        # map english words for numbers to digits
        word2num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        if head in word2num:
            head = word2num[head]

        # normalize some common color variants
        color_map = {
            "grey": "gray",
            "light grey": "gray",
            "dark grey": "gray",
        }
        if ans in color_map:
            head = color_map[ans]
        elif head in color_map:
            head = color_map[head]

        return head

    ans = ""
    exp = ""

    # Try to parse out the explicit "Answer:" and "Explanation:" sections.
    m_ans = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    m_exp = re.search(r"explanation\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if m_ans:
        ans_raw = m_ans.group(1)
        # stop at the next "Explanation:" (if present) to avoid swallowing it
        ans_raw = re.split(r"\n\s*explanation\s*:", ans_raw, flags=re.IGNORECASE)[0]
        ans = normalize_answer(ans_raw)

    # Fallback: sometimes the model may write "The answer is X" without a label.
    if not ans:
        m_alt = re.search(
            r"(?:the\s+answer\s+is|answer)\s+([a-z0-9\-]+)",
            text,
            re.IGNORECASE,
        )
        if m_alt:
            ans = normalize_answer(m_alt.group(1))

    if m_exp:
        exp = m_exp.group(1).strip()
    else:
        exp = text.strip()

    return ans, exp


def load_base_processor_and_model(device):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
    ).to(device)

    return processor, model


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(
    data_root: str,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    max_train_samples: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_csv = os.path.join(data_root, "train_labels.csv")
    train_img_dir = os.path.join(data_root, "train")

    train_df = pd.read_csv(train_csv)
    if max_train_samples is not None and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    processor, base_model = load_base_processor_and_model(device)
    model = apply_lora(base_model)

    # 冻结 base model，只训练 LoRA
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    train_dataset = CLEVRXTrainDataset(train_df, train_img_dir, processor)
    collator = QwenVLDataCollator(processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    print("Samples:", len(train_dataset))
    print("Steps per epoch:", num_update_steps_per_epoch)
    print("Total train steps:", max_train_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=max_train_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        optimizer.zero_grad()

        for step, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | Loss {running_loss / (step+1):.4f}"
            )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Saved LoRA model to", output_dir)


def load_for_inference(output_dir: str, device):
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print("Loading LoRA adapter from", output_dir)
        model = PeftModel.from_pretrained(base_model, output_dir)
    else:
        print("No LoRA adapter found in", output_dir, "- using base model only.")
        model = base_model

    model.eval()
    return processor, model


def generate_answer_once(
    image_path: str,
    question: str,
    processor,
    model,
    device,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.4,
    top_p: float = 0.9,
    seed: int = None,
):
    """
    对单个 (image, question) 做一次生成。
    返回:
        ans: 解析后的短答案（小写）
        exp: 模型生成的解释全文
        output_text: 原始生成文本
    """
    image = Image.open(image_path).convert("RGB")

    user_text = (
        "You are a visual reasoning assistant for synthetic CLEVR-style scenes.\n"
        "Carefully look at the image and answer the question.\n\n"
        f"Question: {question}\n\n"
        "First give a short answer, then a detailed reasoning explanation.\n"
        "Answer MUST be a single word or integer token from the dataset vocabulary.\n"
        "Do NOT add units or extra words in the Answer line.\n"
        "Format:\n"
        "Answer: <short answer>\n"
        "Explanation: <step-by-step reasoning>"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
    # Qwen2-VL 不支持 generator=，改用全局 seed
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )
    else:
        generated_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    output_text = processor.batch_decode(
        gen_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    ans, exp = parse_answer_explanation(output_text)
    return ans, exp, output_text


def run_inference(
    data_root: str,
    output_dir: str,
    submission_path: str,
    ensemble_size: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.4,
    top_p: float = 0.9,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_csv = os.path.join(data_root, "test_non_labels.csv")
    test_img_dir = os.path.join(data_root, "test")

    test_df = pd.read_csv(test_csv)

    processor, model = load_for_inference(output_dir, device)

    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sample_id = row["id"]
        img_path = os.path.join(test_img_dir, row["file"])
        question = row["question"]

        answers = []
        explanations = []

        # ensemble 多次生成
        for k in range(max(1, ensemble_size)):
            do_sample = ensemble_size > 1
            seed = 42 + k  # 固定 seed，保证可复现
            ans_k, exp_k, _ = generate_answer_once(
                img_path,
                question,
                processor,
                model,
                device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            answers.append(ans_k)
            explanations.append(exp_k)

        # 对答案做多数表决（忽略空字符串）
        valid_answers = [a for a in answers if isinstance(a, str) and a.strip() != ""]
        if len(valid_answers) == 0:
            final_ans = answers[0] if answers else ""
            final_exp = explanations[0] if explanations else ""
        else:
            counter = Counter(valid_answers)
            # 先按出现次数降序，再按第一次出现的顺序优先
            final_ans = None
            best_count = -1
            best_index = len(answers)
            for a, c in counter.items():
                first_idx = answers.index(a)
                if c > best_count or (c == best_count and first_idx < best_index):
                    best_count = c
                    best_index = first_idx
                    final_ans = a

            # 选一个与最终答案匹配的 explanation（优先非空的）
            final_exp = ""
            for a, e in zip(answers, explanations):
                if a == final_ans and isinstance(e, str) and e.strip():
                    final_exp = e
                    break
            if not final_exp:
                final_exp = explanations[0] if explanations else ""

        results.append(
            {
                "id": sample_id,
                "answer": final_ans,
                "explanation": final_exp,
            }
        )

    submission = pd.DataFrame(results)[["id", "answer", "explanation"]]
    submission.to_csv(submission_path, index=False)
    print("Saved submission to", submission_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./custom_dataset/custom_dataset",
        help="Path to custom_dataset/custom_dataset directory (contains train/test/*.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen2vl_clevr_lora",
        help="Directory to save LoRA weights.",
    )
    parser.add_argument(
        "--submission_path",
        type=str,
        default="./submission.csv",
        help="Where to write submission.csv.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If set, run LoRA training before inference.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Train batch size.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional: limit number of training samples for low resources.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of generations per sample for ensemble voting at inference.",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate for each sample.",
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=0.4,
        help="Sampling temperature when do_sample=True.",
    )
    parser.add_argument(
        "--gen_top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter when do_sample=True.",
    )

    args = parser.parse_args()

    if args.do_train:
        train(
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            max_train_samples=args.max_train_samples,
        )

    run_inference(
        data_root=args.data_root,
        output_dir=args.output_dir,
        submission_path=args.submission_path,
        ensemble_size=args.ensemble_size,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
    )


if __name__ == "__main__":
    main()
