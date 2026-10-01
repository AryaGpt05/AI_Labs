"""
CS-F407: Artificial Intelligence
Laboratory - Bayesian Networks and Autoregressive Language Models

Author: Arya Gupta
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

TRAINING_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


class FirstOrderLanguageModel:
    """First-order Markov Autoregressive Language Model: P(Xt | Xt-1)"""
    
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.cpt: Dict[str, Dict[str, float]] = {}
        
    def train(self, sentences: List[str]):
        tokenized = [["<START>"] + s.lower().split() + ["<END>"] for s in sentences]
        for s in tokenized:
            for i in range(len(s) - 1):
                self.transitions[s[i]][s[i+1]] += 1
                
        for w, next_counts in self.transitions.items():
            total = sum(next_counts.values())
            self.cpt[w] = {nxt: cnt / total for nxt, cnt in next_counts.items()}
            
    def get_distribution(self, word: str) -> Dict[str, float]:
        return self.cpt.get(word, {})
        
    def predict_most_probable(self, word: str) -> str:
        dist = self.get_distribution(word)
        if not dist:
            return "<END>"
        return max(dist.items(), key=lambda x: x[1])[0]
        
    def generate_sentence(self, greedy: bool = False, max_len: int = 25) -> str:
        curr = "<START>"
        tokens = []
        for _ in range(max_len):
            dist = self.get_distribution(curr)
            if not dist:
                break
            if greedy:
                curr = max(dist.items(), key=lambda x: x[1])[0]
            else:
                words, probs = zip(*dist.items())
                curr = random.choices(words, weights=probs, k=1)[0]
            if curr == "<END>":
                break
            tokens.append(curr)
        return " ".join(tokens)


class SecondOrderLanguageModel:
    """Second-order Autoregressive Language Model: P(Xt | Xt-2, Xt-1)"""
    
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.cpt: Dict[Tuple[str, str], Dict[str, float]] = {}
        
    def train(self, sentences: List[str]):
        tokenized = [["<START>"] + s.lower().split() + ["<END>"] for s in sentences]
        for s in tokenized:
            for i in range(len(s) - 2):
                self.transitions[(s[i], s[i+1])][s[i+2]] += 1
                
        for ctx, next_counts in self.transitions.items():
            total = sum(next_counts.values())
            self.cpt[ctx] = {nxt: cnt / total for nxt, cnt in next_counts.items()}
            
    def get_distribution(self, context: Tuple[str, str]) -> Dict[str, float]:
        return self.cpt.get(context, {})
        
    def generate_sentence(self, greedy: bool = False, max_len: int = 25) -> str:
        w1, w2 = "<START>", "the"
        tokens = [w2]
        for _ in range(max_len):
            dist = self.get_distribution((w1, w2))
            if not dist:
                break
            if greedy:
                nxt = max(dist.items(), key=lambda x: x[1])[0]
            else:
                words, probs = zip(*dist.items())
                nxt = random.choices(words, weights=probs, k=1)[0]
            if nxt == "<END>":
                break
            tokens.append(nxt)
            w1, w2 = w2, nxt
        return " ".join(tokens)


def main():
    print("=" * 65)
    print("Laboratory 5: Bayesian Networks and Language Models")
    print("=" * 65)
    
    # 1. Train First-Order Model
    lm1 = FirstOrderLanguageModel()
    lm1.train(TRAINING_CORPUS)
    
    print("\n--- Part IV: First-Order CPT Tables ---")
    for word in ['<START>', 'the', 'cat', 'dog', 'sat', 'ran']:
        print(f"P(next | '{word}'): {lm1.get_distribution(word)}")
        
    print("\n--- Part VII: Probability Normalization Verification ---")
    for word, dist in lm1.cpt.items():
        total = sum(dist.values())
        print(f"  Word: {word:<10} | Sum of Probabilities: {total:.4f}")
        
    # Generate 20 sentences for First Order Model
    random.seed(42)
    sents_1st = [lm1.generate_sentence(greedy=False) for _ in range(20)]
    with open('Lab5_Bayesian_Networks/generated_sentences_first_order.txt', 'w', encoding='utf-8') as f:
        for idx, s in enumerate(sents_1st, 1):
            f.write(f"{idx:2d}. {s}\n")
    print("\nSaved 20 first-order sentences to generated_sentences_first_order.txt")
    
    # 2. Train Second-Order Model
    lm2 = SecondOrderLanguageModel()
    lm2.train(TRAINING_CORPUS)
    
    print("\n--- Part XI: Second-Order CPT Tables ---")
    contexts_to_show = [('<START>', 'the'), ('the', 'cat'), ('the', 'dog'), ('cat', 'sat'), ('dog', 'ran'), ('on', 'the'), ('to', 'the')]
    for ctx in contexts_to_show:
        print(f"P(next | {str(ctx):<20}): {lm2.get_distribution(ctx)}")
        
    # Generate 20 sentences for Second Order Model
    random.seed(42)
    sents_2nd = [lm2.generate_sentence(greedy=False) for _ in range(20)]
    with open('Lab5_Bayesian_Networks/generated_sentences_second_order.txt', 'w', encoding='utf-8') as f:
        for idx, s in enumerate(sents_2nd, 1):
            f.write(f"{idx:2d}. {s}\n")
    print("Saved 20 second-order sentences to generated_sentences_second_order.txt")


if __name__ == '__main__':
    main()
