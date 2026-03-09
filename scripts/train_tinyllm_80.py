from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model import AdvancedTokenizer, TinyLLM


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<EOS>"]


@dataclass
class Config:
    data_dir: Path = Path("aclImdb")
    output_path: Path = Path("tinyllm_complete.pt")
    vocab_size: int = 18000
    max_len: int = 256
    batch_size: int = 32
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.02
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.2
    seed: int = 42
    workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_imdb_split(split_dir: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for label_name, label_idx in [("neg", 0), ("pos", 1)]:
        dir_path = split_dir / label_name
        for file_path in sorted(dir_path.glob("*.txt")):
            texts.append(file_path.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label_idx)
    return texts, labels


def build_vocab(tokenizer: AdvancedTokenizer, texts: Iterable[str], vocab_size: int) -> None:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenizer._tokenize(text))

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for token, _ in counter.most_common(max(0, vocab_size - len(SPECIAL_TOKENS))):
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer.word2idx = vocab
    tokenizer.idx2word = {idx: token for token, idx in vocab.items()}


class EncodedIMDBDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AdvancedTokenizer, max_len: int):
        self.labels = labels
        self.input_ids: list[list[int]] = []
        self.attn_masks: list[list[int]] = []
        for text in texts:
            ids, mask = tokenizer.encode(text, max_length=max_len)
            self.input_ids.append(ids)
            self.attn_masks.append(mask)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.attn_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def run_epoch(
    model: TinyLLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = tqdm(loader, desc="Train", leave=False)
    for input_ids, attention_mask, labels in progress:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*total_correct/max(total_count,1):.2f}%")
    return total_loss / max(total_count, 1), 100.0 * total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: TinyLLM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    progress = tqdm(loader, desc="Eval ", leave=False)
    for input_ids, attention_mask, labels in progress:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = 100.0 * accuracy_score(all_labels, all_preds)
    macro_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(cfg.__dict__, default=str, indent=2)}")

    tokenizer = AdvancedTokenizer(vocab_size=cfg.vocab_size)
    train_texts, train_labels = read_imdb_split(cfg.data_dir / "train")
    test_texts, test_labels = read_imdb_split(cfg.data_dir / "test")
    print(f"Loaded IMDB: train={len(train_texts)} test={len(test_texts)}")

    print("Building vocabulary...")
    build_vocab(tokenizer, train_texts, cfg.vocab_size)
    print(f"Vocab size used: {len(tokenizer.word2idx)}")

    print("Encoding datasets...")
    train_ds = EncodedIMDBDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    test_ds = EncodedIMDBDataset(test_texts, test_labels, tokenizer, cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False,
    )

    model_config = {
        "vocab_size": len(tokenizer.word2idx),
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "d_ff": cfg.d_ff,
        "max_len": cfg.max_len,
        "num_classes": 2,
        "dropout": cfg.dropout,
        "pad_idx": tokenizer.word2idx["<PAD>"],
        "use_pre_ln": True,
    }
    model = TinyLLM(**model_config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)

    best_test_acc = 0.0
    best_test_f1 = 0.0
    history: dict[str, list[float]] = {"train_acc": [], "test_acc": [], "test_f1": []}

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f}%")

        improved = test_acc > best_test_acc
        if improved:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "tokenizer_word2idx": tokenizer.word2idx,
                    "tokenizer_idx2word": tokenizer.idx2word,
                    "class_names": ["Negative", "Positive"],
                    "history": history,
                    "best_test_acc": best_test_acc,
                    "best_test_f1": best_test_f1,
                },
                cfg.output_path,
            )
            print(f"Saved checkpoint: {cfg.output_path} (best test acc: {best_test_acc:.2f}%)")

    print("\nTraining complete.")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Best test macro-F1: {best_test_f1:.2f}%")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train TinyLLM for >=80% IMDB accuracy.")
    p.add_argument("--data-dir", type=Path, default=Config.data_dir)
    p.add_argument("--output", type=Path, default=Config.output_path)
    p.add_argument("--vocab-size", type=int, default=Config.vocab_size)
    p.add_argument("--max-len", type=int, default=Config.max_len)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--d-model", type=int, default=Config.d_model)
    p.add_argument("--num-heads", type=int, default=Config.num_heads)
    p.add_argument("--num-layers", type=int, default=Config.num_layers)
    p.add_argument("--d-ff", type=int, default=Config.d_ff)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--workers", type=int, default=Config.workers)
    args = p.parse_args()
    return Config(
        data_dir=args.data_dir,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    train(parse_args())
