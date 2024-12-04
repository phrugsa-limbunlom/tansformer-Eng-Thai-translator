from typing import Any, Optional

import tltk


class Tokenizer:
    """
        sentence -> word(token) -> index
    """
    def __init__(self,
                 tokenizer_en: Optional[object] = None,
                 tokenizer_th: Optional[object] = None) \
            -> None:
        self.tokenizer_en = tokenizer_en
        self.tokenizer_th = tokenizer_th

    @staticmethod
    def _build_vocab(vocab):
        vocab = list(set(vocab))
        vocab.sort()
        return {word: i for i, word in enumerate(vocab)}  # dictionary of word and index

    @staticmethod
    def _remove_punctuation(text):
        for punc in ["!", ".", "?"]:
            text = text.replace(punc, "")
        return text

    def _custom_en_tokenize(self, en_text):

        words = [token.lower() for token in self._remove_punctuation(en_text).split(" ")]

        stoi = self._build_vocab(words)

        return [stoi[word] for word in words]

    def _custom_th_tokenize(self, th_text):

        words = [word[0] for piece in tltk.nlp.pos_tag(self._remove_punctuation(th_text)) for word in piece]

        stoi = self._build_vocab(words)

        return [stoi[word] for word in words]

    def _tokenize(self, en_text, th_text):
        if self.tokenizer_en is None and self.tokenizer_th is None:
            return self._custom_en_tokenize(en_text), self._custom_th_tokenize(th_text)

        en_token = self.tokenizer_en(en_text)
        th_token = self.tokenizer_th(th_text)
        return en_token["input_ids"], th_token["input_ids"]

    def preprocessing(self, data: []) -> tuple[list[Any], list[Any], int, int]:
        src_dataset = []
        target_dataset = []

        for en_text, th_text in zip(data["en_text"], data["th_text"]):
            tokenized_en, tokenized_th = self._tokenize(en_text, th_text)
            src_dataset.append(tokenized_en)
            target_dataset.append(tokenized_th)

        vocab_size_en = len(set([token for s in src_dataset for token in s]))
        vocab_size_th = len(set([token for t in target_dataset for token in t]))

        return src_dataset, target_dataset, vocab_size_en, vocab_size_th
