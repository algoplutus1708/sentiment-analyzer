import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Author: Swastick
INFERENCE_VERSION = "tinyllm-v2-high-accuracy"


class AdvancedTokenizer:
    """Advanced tokenizer with special tokens"""

    def __init__(self, vocab_size=15000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<EOS>']

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9'!?.,\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str):
        text = self._normalize(text)
        tokens = text.split()
        result = []
        for token in tokens:
            if token in ['!', '?', ',', '.', "'"]:
                if result and result[-1] not in ['<CLS>', '<SEP>', '<EOS>']:
                    result[-1] = result[-1] + token
                else:
                    result.append(token)
            else:
                result.append(token)
        return [t for t in result if t]

    def encode(self, text: str, max_length=256):
        cls_idx = self.word2idx.get("<CLS>", 2)
        eos_idx = self.word2idx.get("<EOS>", 4)
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
        self.scale = math.sqrt(self.d_k)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        out, weights = self.scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.output_dropout(self.W_o(out))
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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_pre_ln=True):
        super().__init__()
        self.use_pre_ln = use_pre_ln
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.use_pre_ln:
            attn_input = self.norm1(x)
            attn_output, attn_weights = self.attention(attn_input, mask)
            x = x + self.dropout(attn_output)
            
            ff_input = self.norm2(x)
            ff_output = self.ff(ff_input)
            x = x + self.dropout(ff_output)
        else:
            attn_output, attn_weights = self.attention(x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.ff(x)
            x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights


class TinyLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=384,
        num_heads=8,
        num_layers=6,
        d_ff=1536,
        max_len=256,
        num_classes=2,
        dropout=0.15,
        pad_idx=0,
        use_pre_ln=True,
        use_legacy_head=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_pre_ln = use_pre_ln
        self.use_legacy_head = use_legacy_head
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, use_pre_ln) for _ in range(num_layers)]
        )

        if self.use_legacy_head:
            # Backward-compatible classifier for older checkpoints.
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            self.final_norm = nn.LayerNorm(d_model)
            # Multi-pooling classifier
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 4, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            attention_mask = (x != self.pad_idx).long()

        padding_mask = attention_mask.bool()

        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)

        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, padding_mask)
            attention_weights.append(attn_weights)

        cls_repr = x[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_repr = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_repr = sum_repr / count

        if self.use_legacy_head:
            combined = torch.cat([cls_repr, mean_repr], dim=-1)
            logits = self.classifier(combined)
            return logits, attention_weights

        x = self.final_norm(x)

        # Multi-pooling
        x_masked = x.clone()
        x_masked[~padding_mask] = float('-inf')
        max_repr, _ = x_masked.max(dim=1)

        last_attn = attention_weights[-1]
        attn_weights_mean = last_attn.mean(dim=1)
        attn_pool = torch.bmm(attn_weights_mean, x)
        attn_pool_repr = attn_pool[:, 0, :]
        
        combined = torch.cat([cls_repr, mean_repr, max_repr, attn_pool_repr], dim=-1)
        logits = self.classifier(combined)
        return logits, attention_weights


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict
    positive_reply: str
    negative_reply: str


class SentimentPredictor:
    def __init__(self, checkpoint_path="tinyllm_complete.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", ["Negative", "Positive"])
        self.model_config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]

        self.tokenizer = AdvancedTokenizer()
        self.tokenizer.word2idx = checkpoint["tokenizer_word2idx"]
        raw_idx2word = checkpoint["tokenizer_idx2word"]
        if raw_idx2word and isinstance(next(iter(raw_idx2word.keys())), str):
            self.tokenizer.idx2word = {int(k): v for k, v in raw_idx2word.items()}
        else:
            self.tokenizer.idx2word = raw_idx2word

        uses_legacy_head = (
            "final_norm.weight" not in state_dict
            and "classifier.7.weight" not in state_dict
            and "classifier.0.weight" in state_dict
            and state_dict["classifier.0.weight"].ndim == 1
        )
        self.model = TinyLLM(**self.model_config, use_legacy_head=uses_legacy_head)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.max_len = self.model_config.get("max_len", 256)
        self.inference_version = (
            f"{INFERENCE_VERSION}-legacy-compatible" if uses_legacy_head else INFERENCE_VERSION
        )
        self.greeting_words = {
            "hi", "hello", "hey", "yo", "sup", "what's", "whats", "up", "good", "morning", "evening"
        }
        self.sentiment_words = {
            "good", "great", "excellent", "amazing", "love", "awesome", "best",
            "bad", "terrible", "awful", "hate", "worst", "sad", "happy",
            "recommend", "disappointed", "boring", "fantastic", "wonderful",
            "frustrating", "confusing", "slow", "buggy", "broken", "annoying", "poor",
            "beautiful", "strong", "impressive", "watchable", "enjoyable",
            "disappointing", "uneven", "dragging", "long"
        }
        self.positive_words = {
            "good", "great", "excellent", "amazing", "love", "awesome", "best", "happy",
            "recommend", "fantastic", "wonderful", "smooth", "fast", "clean", "simple", "useful",
            "beautiful", "strong", "impressive", "watchable", "enjoyable", "masterpiece"
        }
        self.negative_words = {
            "bad", "terrible", "awful", "hate", "worst", "sad", "disappointed", "boring",
            "frustrating", "confusing", "slow", "buggy", "broken", "annoying", "poor",
            "laggy", "difficult", "hard", "issues", "problem", "problems", "error", "errors",
            "disappointing", "uneven", "dragging", "long", "waste"
        }
        self.negation_words = {"not", "no", "never", "hardly", "rarely", "n't"}
        self.clause_shift_words = {"but", "however", "though", "although", "yet"}

    def _build_replies(self, text: str, predicted_label: str, confidence: float) -> tuple[str, str]:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            cleaned = "your input"
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rstrip() + "..."

        label = predicted_label.lower()
        if label == "positive":
            positive_reply = (
                f"That sounds positive overall. Thanks for sharing: \"{cleaned}\"."
            )
            negative_reply = (
                "A negative interpretation could be that some details might still feel disappointing."
            )
        elif label == "negative":
            positive_reply = (
                "A positive interpretation could be that this can still improve with the right changes."
            )
            negative_reply = (
                f"That sounds frustrating and negative overall: \"{cleaned}\"."
            )
        else:
            positive_reply = (
                f"A positive take could be that this is neutral but still potentially good: \"{cleaned}\"."
            )
            negative_reply = (
                "A negative take could be that the statement is uncertain and may hide dissatisfaction."
            )

        if confidence >= 0.9:
            positive_reply += " (High confidence)"
            negative_reply += " (High confidence)"
        return positive_reply, negative_reply

    def _is_non_sentiment_text(self, text: str) -> bool:
        tokens = [t for t in self.tokenizer._tokenize(text) if t not in {".", ",", "!", "?"}]
        if not tokens:
            return True

        joined = " ".join(tokens)
        has_question = "?" in text or joined.startswith(("what", "how", "why", "when", "where"))
        sentiment_hits = sum(1 for t in tokens if t in self.sentiment_words)
        greeting_hits = sum(1 for t in tokens if t in self.greeting_words)

        if len(tokens) <= 4 and sentiment_hits == 0:
            return True
        if greeting_hits > 0 and sentiment_hits == 0 and len(tokens) <= 7:
            return True
        if has_question and sentiment_hits == 0 and len(tokens) <= 8:
            return True
        return False

    def _normalize_probs(self, neg: float, pos: float) -> tuple[float, float]:
        neg = max(0.0, float(neg))
        pos = max(0.0, float(pos))
        total = neg + pos
        if total <= 1e-8:
            return 0.5, 0.5
        return neg / total, pos / total

    def _apply_lexicon_correction(self, text: str, probs: list[float]) -> tuple[float, float]:
        if len(probs) < 2:
            return 0.5, 0.5

        tokens = [t for t in self.tokenizer._tokenize(text) if t not in {".", ",", "!", "?"}]
        if not tokens:
            return self._normalize_probs(probs[0], probs[1])
        
        token_text = " ".join(tokens)

        # Handle negation patterns
        if re.search(r"\bnot\s+(bad|terrible|awful|worst|disappointing|frustrating)\b", token_text):
            return self._normalize_probs(probs[0], probs[1] + 1.0)
        if re.search(r"\bnot\s+(good|great|excellent|amazing|awesome|wonderful)\b", token_text):
            return self._normalize_probs(probs[0] + 1.0, probs[1])

        neg_score = 0.0
        pos_score = 0.0
        split_idx = -1
        for i, tok in enumerate(tokens):
            if tok in self.clause_shift_words:
                split_idx = i
                break

        for i, tok in enumerate(tokens):
            negated = any(tokens[j] in self.negation_words for j in range(max(0, i - 3), i))
            weight = 1.35 if split_idx >= 0 and i > split_idx else 1.0

            if tok in self.positive_words:
                if negated:
                    neg_score += 2.2 * weight
                else:
                    pos_score += 1.0 * weight
            elif tok in self.negative_words:
                if negated:
                    pos_score += 2.4 * weight
                else:
                    neg_score += 1.2 * weight

        if neg_score >= 1.2 and pos_score == 0.0:
            neg = probs[0] + 1.2 + 0.25 * neg_score
            pos = probs[1]
            return self._normalize_probs(neg, pos)

        neg = probs[0] + 0.18 * neg_score
        pos = probs[1] + 0.18 * pos_score
        return self._normalize_probs(neg, pos)

    def _should_mark_mixed(self, text: str, neg_prob: float, pos_prob: float) -> bool:
        tokens = [t for t in self.tokenizer._tokenize(text) if t not in {".", ",", "!", "?"}]
        if len(tokens) < 16:
            return False

        pos_hits = sum(1 for t in tokens if t in self.positive_words)
        neg_hits = sum(1 for t in tokens if t in self.negative_words)
        if pos_hits < 2 or neg_hits < 2:
            return False

        has_contrast = any(t in self.clause_shift_words for t in tokens)
        prob_gap = abs(pos_prob - neg_prob)
        return has_contrast and prob_gap <= 0.6

    def predict(self, text: str) -> Prediction:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        tokens, mask = self.tokenizer.encode(text, max_length=self.max_len)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        
        neg_prob, pos_prob = self._apply_lexicon_correction(text, probs)
        probs = [neg_prob, pos_prob]

        if self._is_non_sentiment_text(text):
            neg = 0.1 * probs[0]
            pos = 0.1 * probs[1] if len(probs) > 1 else 0.0
            neutral = 1.0 - (neg + pos)
            probabilities = {
                "Negative": float(neg),
                "Positive": float(pos),
                "Neutral": float(neutral),
            }
            positive_reply, negative_reply = self._build_replies(text, "Neutral", float(neutral))
            return Prediction(
                label="Neutral",
                confidence=float(neutral),
                probabilities=probabilities,
                positive_reply=positive_reply,
                negative_reply=negative_reply,
            )

        if self._should_mark_mixed(text, probs[0], probs[1]):
            balance = 1.0 - abs(probs[1] - probs[0])
            neutral = min(0.7, max(0.34, 0.25 + 0.45 * balance))
            carry = 1.0 - neutral
            neg = carry * probs[0]
            pos = carry * probs[1]
            probabilities = {
                "Negative": float(neg),
                "Positive": float(pos),
                "Neutral": float(neutral),
            }
            positive_reply, negative_reply = self._build_replies(text, "Neutral", float(neutral))
            return Prediction(
                label="Neutral",
                confidence=float(neutral),
                probabilities=probabilities,
                positive_reply=positive_reply,
                negative_reply=negative_reply,
            )

        best_idx = int(torch.argmax(torch.tensor(probs)).item())
        probabilities = {
            self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
        }
        label = self.class_names[best_idx]
        confidence = float(probs[best_idx])
        positive_reply, negative_reply = self._build_replies(text, label, confidence)

        return Prediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            positive_reply=positive_reply,
            negative_reply=negative_reply,
        )
