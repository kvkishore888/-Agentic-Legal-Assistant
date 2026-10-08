"""Dependency-free BM25-style lexical retrieval."""
from __future__ import annotations
import math
import re
_TOKEN_RE = re.compile(r"(?u)\b\w+\b")

class KeywordIndex:
    def __init__(self):
        self.chunks = []
        self.df = {}
        self.avgdl = 0.0
        self._terms = []

    def add(self, chunks):
        self.chunks = list(chunks)
        self.df = {}
        self._terms = []
        total = 0
        for c in self.chunks:
            terms = _TOKEN_RE.findall(c["text"].lower())
            self._terms.append(terms)
            total += len(terms)
            for term in set(terms):
                self.df[term] = self.df.get(term, 0) + 1
        self.avgdl = total / max(1, len(self.chunks))

    def search(self, query, top_k=5):
        if top_k <= 0:
            return []
        q = _TOKEN_RE.findall(query.lower())
        n = len(self.chunks)
        k1, b = 1.5, 0.75
        scored = []
        for chunk, terms in zip(self.chunks, self._terms):
            dl = len(terms)
            score = 0.0
            for term in q:
                tf = terms.count(term)
                df = self.df.get(term, 0)
                if tf and df:
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(self.avgdl, 1)))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**chunk, "score": float(score)} for score, chunk in scored[:top_k] if score > 0]
