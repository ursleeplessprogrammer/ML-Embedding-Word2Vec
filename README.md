# 🧭 Word2Vec: A Field Guide to Word Embeddings

> **How can a computer learn that “doctor” and “nurse” are related—without being given a dictionary?**  
> Word2Vec learns numerical representations of words by studying the words that appear around them.

## 🗺️ Trail Map

- [What Is Word2Vec?](#-what-is-word2vec)
- [What Is a Word Embedding?](#-what-is-a-word-embedding)
- [How Word2Vec Learns](#-how-word2vec-learns)
- [The Two Word2Vec Architectures](#️-the-two-word2vec-architectures)
- [The Context Window](#-the-context-window)
- [Training Shortcuts](#️-training-shortcuts)
- [What the Vectors Capture](#-what-the-vectors-capture)
- [What Word2Vec Is Used For](#-what-word2vec-is-used-for)
- [Limitations](#️-limitations)
- [Word2Vec and Related Methods](#-word2vec-and-related-methods)
- [Python Example](#-python-example-with-gensim)
- [Evaluating a Model](#-evaluating-a-model)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔦 What Is Word2Vec?

**Word2Vec** is a method for learning numerical representations of words from text. It was introduced by Tomas Mikolov and colleagues in 2013. Its central idea is that words appearing in similar contexts often have related meanings or uses. ([arxiv.org](https://arxiv.org/abs/1301.3781))

For example, if *doctor* and *nurse* frequently appear near words such as *hospital*, *patient*, and *health*, Word2Vec may learn vectors that place them near one another.

Word2Vec is not a dictionary and does not store a hand-written definition for each word. It learns patterns from a text collection, called a **corpus**.

## 🧩 What Is a Word Embedding?

A computer needs numbers to perform calculations. A **word embedding** represents a word as a list of numbers, or vector:

```text
river → [0.18, -0.42, 0.73, ...]
```

Each value is one dimension in the vector. These dimensions are learned from text and usually do not have simple human-readable labels.

Words used in similar ways often have vectors pointing in similar directions. One common way to compare vectors is **cosine similarity**, which measures how similar their directions are.

## 🧠 How Word2Vec Learns

Word2Vec learns by making predictions from surrounding words. It creates training examples from text and adjusts its vectors to improve those predictions.

Its two main architectures are:

1. **Continuous Bag of Words (CBOW):** predicts a word from its surrounding context.
2. **Skip-gram:** predicts surrounding words from a target word.

The model’s prediction task is mainly a way to learn useful word vectors. After training, applications usually use the learned vectors rather than the prediction model itself.

## 🏗️ The Two Word2Vec Architectures

### CBOW — predict the center word

CBOW uses surrounding words to predict a target word.

```text
The cat sat on the ___
```

Given the surrounding words, the model might predict `mat`.

```text
Context words → CBOW → Target word
```

CBOW is generally efficient to train and can work well for common words.

### Skip-gram — predict the surrounding words

Skip-gram uses a target word to predict nearby context words.

```text
The cat sat on the mat
```

If `sat` is the target, the model learns to predict nearby words such as `cat` and `on`.

```text
Target word → Skip-gram → Context words
```

Skip-gram can be useful for learning representations of less frequent words. Which architecture works better depends on the corpus and training settings.

## 🪟 The Context Window

A **context window** sets how many nearby words are considered context.

For this sentence:

```text
The curious cat slept quietly.
```

If `cat` is the target and the window size is 2, the context can include:

```text
The, curious, slept, quietly
```

A smaller window tends to emphasize local relationships and grammatical patterns. A larger window may capture broader topical relationships. The best choice depends on the task.

## ⚙️ Training Shortcuts

### Negative sampling

Predicting a word from a very large vocabulary can be expensive. **Negative sampling** makes training more efficient by teaching the model to distinguish real word-context pairs from a small number of randomly selected pairs.

In simplified terms, it encourages:

- Real pairs, such as `coffee` and `cup`, to have compatible vectors.
- Random negative pairs to have less compatible vectors.

### Hierarchical softmax

**Hierarchical softmax** organizes the vocabulary in a tree. This allows the model to calculate predictions without evaluating every vocabulary word individually. It is an alternative to standard softmax and negative sampling.

### Frequent-word subsampling

Very common words, such as *the* and *is*, may provide less useful information in many contexts. Word2Vec can reduce how often these words are used during training.

## 🔍 What the Vectors Capture

Word2Vec vectors can reflect patterns of meaning and grammar found in the training text. Words that occur in similar contexts often end up near one another in vector space.

You may encounter examples of vector arithmetic such as:

```text
king - man + woman ≈ queen
```

Treat this as an illustration, not a dependable rule. Results vary with the training data and settings. Vector relationships do not mean the model understands concepts the way a person does.

## 🌍 What Word2Vec Is Used For

Word2Vec embeddings can support natural language processing tasks such as:

- Finding words used in similar contexts
- Grouping related terms
- Representing words as numerical features for machine-learning models
- Supporting text classification, including sentiment analysis
- Improving search or matching systems
- Studying word relationships in a specific collection of text

Training on domain-specific material—such as product reviews or scientific papers—can produce vectors that reflect that vocabulary and style.

## ⚠️ Limitations

### One vector per word

Traditional Word2Vec gives each vocabulary word one vector. It does not naturally change that vector based on the sentence.

For example, *bank* has different meanings in:

- “We sat by the river bank.”
- “She deposited money at the bank.”

Word2Vec’s single vector blends its uses in the training data. Contextual language models can create different representations based on the surrounding sentence.

### Limited sentence understanding

A Word2Vec vector represents a word, not a complete sentence. The standard approach does not preserve the full order or structure of a sentence.

### Out-of-vocabulary words

A traditional Word2Vec model cannot provide a learned vector for a word absent from its vocabulary. **FastText** addresses this limitation in part by using character fragments as well as whole words.

### Bias in training data

Embeddings reflect patterns in their training text. If the corpus contains stereotypes or imbalanced representation, the learned vectors may reflect those patterns.

### Results depend on the corpus and settings

The text collection, tokenization, context window, vector size, and training method all affect the result. Two models trained with different data or settings can produce different vectors.

## 🧭 Word2Vec and Related Methods

| Method | Main idea | Context-dependent representations? |
|---|---|---|
| One-hot encoding | Assigns each vocabulary word its own position | No |
| Word2Vec | Learns a dense vector for each word from nearby text | No |
| GloVe | Learns vectors from global word co-occurrence patterns | No |
| FastText | Uses word vectors built partly from character fragments | No |
| Transformer language models | Build representations using surrounding text | Yes |

Word2Vec is a **static embedding** method: a word generally has the same vector in different sentences. Transformer-based models produce **contextual representations**, so a word’s representation can change with its context.

## 🧪 Python Example with Gensim

Gensim provides a Word2Vec implementation with CBOW and Skip-gram options. Its documentation shows how to train a model and access vectors through `model.wv`. ([radimrehurek.com](https://radimrehurek.com/gensim/models/word2vec.html))

Install Gensim:

```bash
pip install gensim
```

Train a small model:

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "near", "the", "cat"],
    ["a", "cat", "chased", "a", "mouse"],
    ["a", "dog", "chased", "a", "ball"],
]

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1,
    sg=1,       # 1 = Skip-gram; 0 = CBOW
    negative=5,
    seed=42,
)

print("Vector for 'cat':")
print(model.wv["cat"])

print("\nWords most similar to 'cat':")
print(model.wv.most_similar("cat", topn=3))
```

This tiny dataset demonstrates the API; it is far too small to produce dependable embeddings.

### Common parameters

- `vector_size`: Number of dimensions in each word vector.
- `window`: Maximum distance between a target word and its context words.
- `min_count`: Ignore words that occur fewer times than this value.
- `sg`: Use `1` for Skip-gram or `0` for CBOW.
- `negative`: Number of negative samples used during training.
- `workers`: Number of worker threads used for training.
- `seed`: Random seed, which can help make experiments repeatable.

## 📏 Evaluating a Model

There is no single score that proves embeddings are good. Evaluate them according to how you intend to use them.

Possible checks include:

- **Nearest neighbors:** Do similar words have sensible neighbors?
- **Similarity tasks:** Do vector similarity scores align with human judgments?
- **Downstream tasks:** Do the embeddings improve the classifier or search system you care about?
- **Bias checks:** Do the vectors encode unwanted associations?
- **Domain review:** Are specialized words represented appropriately?

A model that performs well on a general benchmark may not be the best choice for a specific domain.

## 📝 Field Notes

- Word2Vec learns dense word vectors from patterns in nearby words.
- CBOW predicts a word from its context; Skip-gram predicts context words from a target word.
- Negative sampling and hierarchical softmax can make training more efficient.
- Traditional Word2Vec gives each word a static representation.
- Training data and settings strongly affect the vectors.
- Word2Vec remains a useful way to learn and work with embeddings, even though contextual models suit many language tasks better.

## 🔗 Further Reading

- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) ([arxiv.org](https://arxiv.org/abs/1301.3781))
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html) ([radimrehurek.com](https://radimrehurek.com/gensim/models/word2vec.html))
- [TensorFlow Word2Vec Tutorial](https://www.tensorflow.org/text/tutorials/word2vec) ([tensorflow.org](https://www.tensorflow.org/text/tutorials/word2vec))

---

*Word2Vec is a foundational NLP technique: simple enough to explain, yet powerful enough to reveal useful patterns in large collections of text.*
