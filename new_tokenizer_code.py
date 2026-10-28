import re
import warnings
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


# LAYER 1: Pre-tokenization (Whitespace + Rule-based)

class PreTokenizer:
    TOKEN_PATTERNS = [
        ("CODE_OPERATOR",   r'(==|!=|<=|>=|->|::|//|<<|>>|\+=|-=|\*=|/=|\*\*|&&|\|\|)'),
        ("NUMBER",          r'\b\d+(\.\d+)?\b'),
        ("EMAIL",           r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
        ("URL",             r'https?://[^\s]+'),
        ("HASHTAG",         r'#[A-Za-z0-9_]+'),
        ("MENTION",         r'@[A-Za-z0-9_]+'),
        ("CONTRACTION",     r"[A-Za-z]+'[A-Za-z]+"),           # don't, it's
        ("WORD",            r"[A-Za-z]+(?:-[A-Za-z]+)*"),      # hyphenated words
        ("PUNCTUATION",     r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]'),
        ("WHITESPACE",      r'\s+'),
        ("UNKNOWN",         r'.'),
    ]

    def __init__(self):
        combined = '|'.join(f'(?P<{name}>{pat})' for name, pat in self.TOKEN_PATTERNS)
        self.regex = re.compile(combined)

    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        tokens = []
        for match in self.regex.finditer(text):
            kind = match.lastgroup
            value = match.group()
            if kind != "WHITESPACE":
                tokens.append((kind, value))
        return tokens

# LAYER 2: Vocabulary Builder (BPE-inspired)
SPECIAL_TOKENS = {
    "[PAD]":   0,
    "[UNK]":   1,
    "[CLS]":   2,
    "[SEP]":   3,
    "[MASK]":  4,
    "[CODE]":  5,
    "[NUM]":   6,
    "[URL]":   7,
    "[EMAIL]": 8,
}

_SPECIAL_TOKEN_OFFSET = len(SPECIAL_TOKENS)  # = 9


class VocabularyBuilder:
    """Learns a subword vocabulary using Byte Pair Encoding."""

    def __init__(self, vocab_size: int = 200):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}

    def _get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """Convert corpus into character-split word frequencies."""
        freq = Counter()
        for sentence in corpus:
            for word in sentence.strip().split():
                freq[' '.join(list(word)) + ' </w>'] += 1
        return dict(freq)

    def _get_pairs(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count all adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        new_freqs = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(^| )' + bigram + r'( |$)')
        replacement_str = ''.join(pair)

        for word in word_freqs:
            new_word = pattern.sub(
                lambda m: (m.group(1)) + replacement_str + (m.group(2)),
                word
            )
            new_freqs[new_word] = word_freqs[word]
        return new_freqs

    def fit(self, corpus: List[str]):
        
        word_freqs = self._get_word_freqs(corpus)

        all_chars = set()
        for word in word_freqs:
            for sym in word.split():
                all_chars.add(sym)

        self.vocab = {
            c: i + _SPECIAL_TOKEN_OFFSET
            for i, c in enumerate(sorted(all_chars))
        }
        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            merged = ''.join(best)
            self.vocab[merged] = len(self.vocab) + _SPECIAL_TOKEN_OFFSET
            word_freqs = self._merge_pair(best, word_freqs)

        return self

# LAYER 3: Subword Tokenizer
class SubwordTokenizer:

    def __init__(self, merges: List[Tuple[str, str]]):
        self.merges = {pair: ''.join(pair) for pair in merges}

    def tokenize_word(self, word: str) -> List[str]:
       
        if not word:
            return []

        symbols = list(word) + ['</w>']
        changed = True

        while changed and len(symbols) > 1:
            changed = False
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1:
                    pair = (symbols[i], symbols[i + 1])
                    if pair in self.merges:
                        new_symbols.append(self.merges[pair])
                        i += 2
                        changed = True
                        continue
                
                new_symbols.append(symbols[i])
                i += 1
            symbols = new_symbols

        return symbols

# LAYER 4: Special Token Handler
TYPE_TO_SPECIAL = {
    "URL":    "[URL]",
    "EMAIL":  "[EMAIL]",
    "NUMBER": "[NUM]",
}

# MAIN: HybridTokenizer
class HybridTokenizer:
    def __init__(self, vocab_size: int = 300, normalize: bool = True):
        self.vocab_size = vocab_size
        self.normalize = normalize
        self.pre_tokenizer = PreTokenizer()
        self.vocab_builder = VocabularyBuilder(vocab_size)
        self.subword_tokenizer = None
        self.is_fitted = False

    def fit(self, corpus: List[str]):
        self.vocab_builder.fit(corpus)
        self.subword_tokenizer = SubwordTokenizer(self.vocab_builder.merges)
        self.is_fitted = True
        return self

    def tokenize(self, text: str) -> List[str]:
        if not self.is_fitted:
            warnings.warn(
                "HybridTokenizer.tokenize() called before fit(). "
                "Subword splitting is disabled; words will be lowercased only.",
                UserWarning,
                stacklevel=2,
            )

        pre_tokens = self.pre_tokenizer.tokenize(text)
        final_tokens = []

        for token_type, token_value in pre_tokens:
            
            if self.normalize and token_type in TYPE_TO_SPECIAL:
                final_tokens.append(TYPE_TO_SPECIAL[token_type])
                continue

            if token_type in ("CODE_OPERATOR", "PUNCTUATION", "HASHTAG", "MENTION"):
                final_tokens.append(token_value)
                continue
            if self.is_fitted and token_type in ("WORD", "CONTRACTION"):
                subwords = self.subword_tokenizer.tokenize_word(token_value.lower())
                final_tokens.extend(subwords)
            else:
                final_tokens.append(token_value.lower())

        return final_tokens

    def encode(self, text: str) -> List[int]:
       
        tokens = self.tokenize(text)
        all_vocab = {**SPECIAL_TOKENS, **self.vocab_builder.vocab}
        return [all_vocab.get(t, SPECIAL_TOKENS["[UNK]"]) for t in tokens]

    def get_vocab(self) -> Dict[str, int]:
        return {**SPECIAL_TOKENS, **self.vocab_builder.vocab}

    def vocab_size_actual(self) -> int:
        return len(self.get_vocab())

# DEMO / TEST

if __name__ == "__main__":
    # Training corpus
    corpus = [
        "Natural language processing is a subfield of artificial intelligence.",
        "Tokenization splits text into meaningful units called tokens.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn representations from data.",
        "It's important to handle punctuation, numbers, and special characters!",
        "def tokenize(text): return text.split()",
        "Visit https://example.com for more details.",
        "Contact us at support@example.com",
        "The accuracy was 98.6% on the test set.",
        "Python uses == for equality and != for inequality.",
    ]

    # Initialize and train
    tokenizer = HybridTokenizer(vocab_size=300, normalize=True)
    tokenizer.fit(corpus)

    # Test sentences
    test_inputs = [
        "Hello, world! This is a test.",
        "I don't think it's working correctly.",
        "Check https://github.com/repo for the code.",
        "x += 1 if x >= 0 else x -= 1",
        "The model achieved 95.4% accuracy on NLP tasks!",
        "Send an email to alice@university.edu",
        "#NLP @student it's tokenization time!",
        "def bpe_merge(pair, vocab): return vocab",
    ]

    print("=" * 60)
    print("     CUSTOM HYBRID TOKENIZER — OUTPUT DEMO")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size_actual()}\n")

    for sentence in test_inputs:
        tokens = tokenizer.tokenize(sentence)
        ids    = tokenizer.encode(sentence)
        print(f"Input : {sentence}")
        print(f"Tokens: {tokens}")
        print(f"IDs   : {ids}")
        print(f"Count : {len(tokens)} tokens")
        print("-" * 60)

    # Token type breakdown example
    print("\n--- Pre-tokenizer Type Analysis ---")
    sample = "Send 5 emails to admin@test.com and visit https://nlp.org #NLP x != y"
    pre = PreTokenizer()
    typed_tokens = pre.tokenize(sample)
    for ttype, tval in typed_tokens:
        print(f"  [{ttype:15s}]  →  {tval}")

    # Verify no ID collision between special tokens and BPE vocab
    print("\n--- Vocabulary Collision Check ---")
    vocab = tokenizer.get_vocab()
    special_ids = set(SPECIAL_TOKENS.values())
    bpe_ids = set(tokenizer.vocab_builder.vocab.values())
    collision = special_ids & bpe_ids
    if collision:
        print(f"Collision found at IDs: {collision}")
    else:
        print(f"No ID collisions. Special token IDs: {sorted(special_ids)}")
        print(f"BPE vocab ID range: {min(bpe_ids)} – {max(bpe_ids)}")