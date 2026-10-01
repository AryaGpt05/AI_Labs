# Laboratory 6 Report — Transformers & Modern LLM Architectures

**Course:** CS-F407 Artificial Intelligence  
**Student Name:** Arya Gupta  
**Date:** Semester 1, 2026-2027  

---

## 1. Overview of Transformer Architectures

Modern Natural Language Processing and Generative AI rely on the Transformer architecture introduced by Vaswani et al. (2017). This laboratory explores three foundational paradigms:

1. **Encoder-Decoder Architecture (Sequence-to-Sequence):**
   - *Exemplar Models:* T5 (`google-t5/t5-base`), FLAN-T5 (`rvv-karma/English2Hinglish-Flan-T5-Base`), BART.
   - *Mechanism:* The encoder processes full bidirectional context across the source sequence; the autoregressive decoder attends to both encoder representations (cross-attention) and previously generated target tokens (causal self-attention).
   - *Applications:* Machine translation, summarization, question answering.

2. **Decoder-Only Architecture (Autoregressive Generative Models):**
   - *Exemplar Models:* GPT-2, GPT-4, LLaMA, Mistral, Gemma.
   - *Mechanism:* Uses causal self-attention masks to predict the next token $P(X_t \mid X_1, \dots, X_{t-1})$ without looking ahead.
   - *Applications:* Open-ended text generation, dialogue, in-context learning.

3. **Encoder-Only Architecture (Bidirectional Representation Models):**
   - *Exemplar Models:* BERT (`bert-base-uncased`), RoBERTa.
   - *Mechanism:* Full bidirectional self-attention across all tokens simultaneously using Masked Language Modeling (MLM).
   - *Applications:* Sequence classification, sentiment analysis, named entity recognition.

---

## 2. Experimental Explorations

### Part A: Sequence-to-Sequence Translation with FLAN-T5
- Input text in English is processed through cross-attention to output target sentences in Hinglish / translated languages.
- Demonstrates how seq2seq models align cross-lingual semantic representations.

### Part B: Text Generation Pipeline with Decoder-Only GPT-2
- Demonstrates temperature-based next-token sampling and prompt continuation.
- Connects directly to the autoregressive chain-rule factorisation studied in Lab 5.

### Part C: Text Classification with BERT
- Sequences are encoded into contextual embeddings; the `[CLS]` token logit is passed to a classification head with Softmax and Cross-Entropy loss.

---

## 3. Comparative Summary

| Architecture Paradigm | Attention Type | Primary Objective | Key Strengths |
| :--- | :--- | :--- | :--- |
| **Encoder-Only (BERT)** | Bidirectional | Masked LM & Classification | Rich deep bidirectional context understanding |
| **Decoder-Only (GPT)** | Causal (Unidirectional) | Autoregressive Next-Token | Coherent, versatile open-ended generation |
| **Encoder-Decoder (T5)** | Bidirectional + Causal | Sequence-to-Sequence Mapping | Robust source-to-target transformation |

---

## 4. Key Takeaways

- **Unified Attention Mechanism:** Self-attention computes $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, allowing tokens to attend dynamically to relevant context regardless of distance.
- **Scaling Laws:** While Bayesian n-gram models suffer from exponential parameter explosion ($|V|^n$), Transformers scale efficiently via distributed continuous parameter representations.