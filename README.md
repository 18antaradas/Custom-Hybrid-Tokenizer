# 🔤 Custom Hybrid Tokenizer

A from-scratch NLP tokenizer that combines **rule-based pre-tokenization**, **Byte Pair Encoding (BPE)-inspired subword learning**, and **special-token normalization** in one layered pipeline. It handles English text, punctuation, contractions, URLs, emails, hashtags, mentions, numbers and basic code snippets, using only the Python standard library.

> NLP project, MCA 4th Semester, Department of Computer Applications.

---

## ✨ Features

- **4-layer architecture:** every stage is separate, readable and easy to swap out
- **No external NLP libraries:** only `re`, `collections` and `typing`
- **Typed pre-tokenization:** 11 token categories matched in priority order
- **BPE learned from your corpus:** with a `</w>` end-of-word marker to preserve word boundaries
- **Special-token normalization:** URLs, emails and numbers become `[URL]`, `[EMAIL]`, `[NUM]`
- **Atomic code operators:** `==`, `!=`, `>=`, `+=`, `->`, `::` and more stay as single tokens
- **Social-media friendly:** `#hashtags` and `@mentions` kept whole
- **Graceful OOV handling:** unseen words fall back to character-level subwords
- **Encoding support:** maps tokens to integer IDs, with 9 reserved special tokens

## 🏗️ Architecture

```
Raw text
   ↓
Layer 1  PreTokenizer       regex split + token type labels
   ↓
Layer 2  VocabularyBuilder  learns BPE merges from the training corpus
   ↓
Layer 3  SubwordTokenizer   applies merges to split words into subwords
   ↓
Layer 4  HybridTokenizer    normalization, orchestration, encode()
   ↓
Tokens / IDs
```

### Pre-tokenizer categories (in match priority)

| Type | Matches |
|---|---|
| `CODE_OPERATOR` | `==  !=  <=  >=  ->  ::  //  <<  >>  +=  -=  *=  /=  **  &&  \|\|` |
| `NUMBER` | Integers and decimals |
| `EMAIL` | Standard email addresses |
| `URL` | `http://` and `https://` links |
| `HASHTAG` | `#` followed by alphanumerics |
| `MENTION` | `@` followed by alphanumerics |
| `CONTRACTION` | Apostrophe-joined words (`don't`, `it's`) |
| `WORD` | Words, including hyphenated ones |
| `PUNCTUATION` | Standard punctuation |
| `WHITESPACE` | Discarded |
| `UNKNOWN` | Any other character |

### Special tokens

| Token | ID | Token | ID |
|---|---|---|---|
| `[PAD]` | 0 | `[CODE]` | 5 |
| `[UNK]` | 1 | `[NUM]` | 6 |
| `[CLS]` | 2 | `[URL]` | 7 |
| `[SEP]` | 3 | `[EMAIL]` | 8 |
| `[MASK]` | 4 | | |

## 🚀 Getting Started

Requires Python 3.7+. No installation needed.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python new_tokenizer_code.py
```

### Usage

```python
from new_tokenizer_code import HybridTokenizer

corpus = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Visit https://example.com for more details.",
    "Python uses == for equality and != for inequality.",
]

tokenizer = HybridTokenizer(vocab_size=300, normalize=True)
tokenizer.fit(corpus)

text = "Check https://github.com/repo for the code."
print(tokenizer.tokenize(text))
# ['ch', 'e', 'c', 'k', '</w>', '[URL]', 'fo', 'r', '</w>', 'the</w>', ...]

print(tokenizer.encode(text))     # integer IDs
print(tokenizer.get_vocab())      # token -> ID mapping
```

Set `normalize=False` to keep URLs, emails and numbers as lowercased raw tokens instead of placeholders.

## 📋 Example Output

Trained on the 10-sentence demo corpus with `vocab_size=300` (actual vocabulary: **309** tokens, including the 9 special tokens).

| Input | Tokens |
|---|---|
| `Send an email to alice@university.edu` | `[..., 'to</w>', '[EMAIL]']` |
| `x += 1 if x >= 0 else x -= 1` | `['x', '</w>', '+=', '[NUM]', 'i', 'f</w>', 'x', '</w>', '>=', '[NUM]', ...]` |
| `I don't think it's working correctly.` | `[..., 'do', 'n', "'", 't</w>', ...]` |
| `#NLP @student it's tokenization time!` | `['#NLP', '@student', 'it', "'", 's</w>', 'token', 'i', 'z', 'ati', 'on</w>', ...]` |

Pre-tokenizer type analysis of `Send 5 emails to admin@test.com and visit https://nlp.org #NLP x != y`:

```
[WORD          ] → Send
[NUMBER        ] → 5
[EMAIL         ] → admin@test.com
[URL           ] → https://nlp.org
[HASHTAG       ] → #NLP
[CODE_OPERATOR ] → !=
```

## ⚠️ Limitations

- BPE is trained on a tiny demo corpus, so merges are not optimal; use a larger domain corpus for real work
- Operators, hashtags and mentions not present in the vocabulary encode to `[UNK]` (ID 1)
- A standalone `</w>` token can appear in the output where a word ends without a merge
- English only; no Hindi/Devanagari or multilingual support
- No detokenizer to rebuild the original text

## 🔮 Future Improvements

- Detokenizer (tokens back to text)
- Hindi and multilingual support with Unicode-aware rules
- Training on larger domain-specific corpora
- WordPiece likelihood criterion as an alternative to frequency-based BPE
- Token position and span offsets for downstream tasks
- Add operators, hashtags and mentions to the vocabulary to avoid `[UNK]`

## 📚 References

- Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
- Devlin et al. (2019). *BERT.* NAACL.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
- Kudo & Richardson (2018). *SentencePiece.* EMNLP.
- Jurafsky & Martin (2023). *Speech and Language Processing* (3rd ed. draft).

## 👥 Authors

*(Add your names and roll numbers here.)*

## 📄 License

For academic and educational purposes.
