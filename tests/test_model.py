import unittest

from app.model import AdvancedTokenizer, SentimentPredictor


class AdvancedTokenizerTests(unittest.TestCase):
    def test_encode_adds_special_tokens_and_padding(self):
        tokenizer = AdvancedTokenizer()
        tokenizer.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<CLS>": 2,
            "<SEP>": 3,
            "<EOS>": 4,
            "great": 5,
            "movie!": 6,
        }

        token_ids, attention_mask = tokenizer.encode("Great movie!", max_length=6)

        self.assertEqual(token_ids, [2, 5, 6, 4, 0, 0])
        self.assertEqual(attention_mask, [1, 1, 1, 1, 0, 0])


class SentimentPredictorHelperTests(unittest.TestCase):
    def test_non_sentiment_greeting_is_detected(self):
        predictor = SentimentPredictor.__new__(SentimentPredictor)
        predictor.tokenizer = AdvancedTokenizer()
        predictor.greeting_words = {
            "hi", "hello", "hey", "yo", "sup", "what's", "whats", "up", "good", "morning", "evening"
        }
        predictor.sentiment_words = {
            "good", "great", "excellent", "amazing", "love", "awesome", "best",
            "bad", "terrible", "awful", "hate", "worst", "sad", "happy",
            "recommend", "disappointed", "boring", "fantastic", "wonderful",
            "frustrating", "confusing", "slow", "buggy", "broken", "annoying", "poor",
            "beautiful", "strong", "impressive", "watchable", "enjoyable",
            "disappointing", "uneven", "dragging", "long"
        }

        self.assertTrue(predictor._is_non_sentiment_text("hello there"))


if __name__ == "__main__":
    unittest.main()
