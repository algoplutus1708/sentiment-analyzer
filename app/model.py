import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Author: Swastick
INFERENCE_VERSION = "tinyllm-neutral-guard-v1"


class SimpleTokenizer:
    def __init__(self, vocab_size=12000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9'!?.,\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str):
        text = self._normalize(text)
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[!?.,]", text)

    def encode(self, text: str, max_length=160):
        cls_idx = self.word2idx.get("<CLS>", 2)
        eos_idx = self.word2idx.get("<EOS>", 3)
        pad_idx = self.word2idx.get("<PAD>", 0)
        unk_idx = self.word2idx.get("<UNK>", 1)

        tokens = ["<CLS>"] + self._tokenize(text) + ["<EOS>"]
        indices = [self.word2idx.get(token, unk_idx) for token in tokens]

        if len(indices) > max_length:
            indices = indices[:max_length]
            indices[-1] = eos_idx

        attention_mask = [1] * len(indices)
        if len(indices) < max_length:
            pad_len = max_length - len(indices)
            indices += [pad_idx] * pad_len
            attention_mask += [0] * pad_len

        return indices, attention_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        out, weights = self.scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)
        return out, weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights


class TinyLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_len=128,
        num_classes=2,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            attention_mask = (x != self.pad_idx).long()

        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()

        for block in self.transformer_blocks:
            x, _ = block(x, attn_mask)

        cls_repr = x[:, 0, :]
        masked_x = x * attention_mask.unsqueeze(-1)
        pooled_repr = masked_x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        features = torch.cat([cls_repr, pooled_repr], dim=-1)
        logits = self.classifier(features)
        return logits


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict


class SentimentPredictor:
    def __init__(self, checkpoint_path="tinyllm_complete.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", ["Negative", "Positive"])
        self.model_config = checkpoint["model_config"]

        self.tokenizer = SimpleTokenizer()
        self.tokenizer.word2idx = checkpoint["tokenizer_word2idx"]
        raw_idx2word = checkpoint["tokenizer_idx2word"]
        if raw_idx2word and isinstance(next(iter(raw_idx2word.keys())), str):
            self.tokenizer.idx2word = {int(k): v for k, v in raw_idx2word.items()}
        else:
            self.tokenizer.idx2word = raw_idx2word

        self.model = TinyLLM(**self.model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.max_len = self.model_config.get("max_len", 160)
        self.inference_version = INFERENCE_VERSION
        self.greeting_words = {
            "hi", "hello", "hey", "yo", "sup", "what's", "whats", "up", "good", "morning", "evening"
        }
        self.sentiment_words = {
            "good", "great", "excellent", "amazing", "love", "awesome", "best",
            "bad", "terrible", "awful", "hate", "worst", "sad", "happy",
            "recommend", "disappointed", "boring", "fantastic", "wonderful"
        }

    def _is_non_sentiment_text(self, text: str) -> bool:
        tokens = [t for t in self.tokenizer._tokenize(text) if t not in {".", ",", "!", "?"}]
        if not tokens:
            return True

        joined = " ".join(tokens)
        has_question = "?" in text or joined.startswith(("what", "how", "why", "when", "where"))
        sentiment_hits = sum(1 for t in tokens if t in self.sentiment_words)
        greeting_hits = sum(1 for t in tokens if t in self.greeting_words)

        # Very short conversational text with no clear sentiment cue.
        if len(tokens) <= 4 and sentiment_hits == 0:
            return True
        # Greetings/questions like "what's up", "hi", "how are you".
        if greeting_hits > 0 and sentiment_hits == 0 and len(tokens) <= 7:
            return True
        if has_question and sentiment_hits == 0 and len(tokens) <= 8:
            return True
        return False

    def predict(self, text: str) -> Prediction:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        tokens, mask = self.tokenizer.encode(text, max_length=self.max_len)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

        if self._is_non_sentiment_text(text):
            # Add a neutral fallback for non-review conversational text.
            neg = 0.1 * probs[0]
            pos = 0.1 * probs[1] if len(probs) > 1 else 0.0
            neutral = 1.0 - (neg + pos)
            probabilities = {
                "Negative": float(neg),
                "Positive": float(pos),
                "Neutral": float(neutral),
            }
            return Prediction(
                label="Neutral",
                confidence=float(neutral),
                probabilities=probabilities,
            )

        best_idx = int(torch.argmax(torch.tensor(probs)).item())
        probabilities = {
            self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
        }

        return Prediction(
            label=self.class_names[best_idx],
            confidence=float(probs[best_idx]),
            probabilities=probabilities,
        )
