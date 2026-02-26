from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 示例语料
corpus = ["Hello world!", "Hello, HuggingFace tokenizers!", "你好，世界！"]

# ===== BPE Tokenizer =====
bpe_tokenizer = Tokenizer(models.BPE())
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer_bpe = trainers.BpeTrainer(
    vocab_size=50,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
bpe_tokenizer.train_from_iterator(corpus, trainer=trainer_bpe)

print("=== BPE Tokenization ===")
for text in corpus:
    print(text, "->", bpe_tokenizer.encode(text).tokens)

# ===== Byte-level BPE Tokenizer =====
bbpe_tokenizer = Tokenizer(models.BPE())
bbpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer_bbpe = trainers.BpeTrainer(
    vocab_size=50,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
bbpe_tokenizer.train_from_iterator(corpus, trainer=trainer_bbpe)

print("\n=== Byte-level BPE Tokenization ===")
for text in corpus:
    print(text, "->", bbpe_tokenizer.encode(text).tokens)
