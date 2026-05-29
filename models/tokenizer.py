import sentencepiece
import torch


class PaligemmaTokenizer(object):
    def __init__(self, max_len=48, path="./ckpts/paligemma_tokenizer.model"):
        self._max_len = max_len
        with open(path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt):
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return torch.tensor(tokens), torch.tensor(mask)
