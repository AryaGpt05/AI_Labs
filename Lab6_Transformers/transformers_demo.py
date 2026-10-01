"""
CS-F407: Artificial Intelligence
Laboratory - Transformers & Modern Language Model Architectures

Author: Arya Gupta
"""

import sys

def explain_transformer_architectures():
    print("=" * 65)
    print("CS-F407 AI: Transformer Architecture Paradigms")
    print("=" * 65)
    
    paradigms = {
        "1. Encoder-Decoder (Sequence-to-Sequence)": {
            "Models": "T5 (google-t5/t5-base), FLAN-T5, BART",
            "Attention": "Bidirectional self-attention in encoder + cross-attention + causal self-attention in decoder",
            "Applications": "Machine translation (English -> Hinglish), text summarization, question answering"
        },
        "2. Decoder-Only (Autoregressive Generative)": {
            "Models": "GPT-2, GPT-4, LLaMA, Mistral, Gemma",
            "Attention": "Causal (unidirectional) masked self-attention",
            "Applications": "Open-ended text completion, code generation, conversational chat"
        },
        "3. Encoder-Only (Bidirectional Representation)": {
            "Models": "BERT (bert-base-uncased), RoBERTa",
            "Attention": "Full bidirectional self-attention",
            "Applications": "Text classification, sentiment analysis, named entity recognition"
        }
    }
    
    for name, info in paradigms.items():
        print(f"\n{name}")
        print("-" * len(name))
        for k, v in info.items():
            print(f"  • {k}: {v}")
            
    print("\n" + "=" * 65)
    print("Core Attention Formulation: Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V")
    print("=" * 65)

if __name__ == '__main__':
    explain_transformer_architectures()
