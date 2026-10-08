"""Dependency-free BM25-style lexical retrieval with metadata filtering."""
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

    @staticmethod
    def _matches(value, condition):
        if not isinstance(condition, dict):
            return value == condition
        for op, expected in condition.items():
            if op == "$eq" and value != expected: return False
            if op == "$ne" and value == expected: return False
            if op == "$in" and value not in expected: return False
            if op == "$nin" and value in expected: return False
        return True

    @classmethod
    def _matches_where(cls, chunk, where):
        if not where:
            return True
        meta = {**(chunk.get("metadata") or {}), **{
            k: chunk.get(k) for k in ("document", "page", "section", "chunk_id")
            if chunk.get(k) is not None
        }}
        if "$and" in where:
            return all(cls._matches_where(chunk, x) for x in where["$and"])
        if "$or" in where:
            return any(cls._matches_where(chunk, x) for x in where["$or"])
        return all(cls._matches(meta.get(key), condition) for key, condition in where.items())

    def search(self, query, top_k=5, where=None):
        if top_k <= 0:
            return []
        q = _TOKEN_RE.findall(query.lower())
        eligible = [
            (chunk, terms) for chunk, terms in zip(self.chunks, self._terms)
            if self._matches_where(chunk, where)
            and str((chunk.get("metadata") or {}).get("content_type", "LEGAL_EVIDENCE"))
                != "EVALUATION_INSTRUCTIONS"
        ]
        n = len(eligible)
        k1, b = 1.5, 0.75
        avgdl = sum(len(terms) for _, terms in eligible) / max(1, n)
        scored = []
        df = {}
        for _, terms in eligible:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        for chunk, terms in eligible:
            dl = len(terms)
            score = 0.0
            for term in q:
                tf = terms.count(term)
                term_df = df.get(term, 0)
                if tf and term_df:
                    idf = math.log(1 + (n - term_df + 0.5) / (term_df + 0.5))
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**chunk, "score": float(score)} for score, chunk in scored[:top_k] if score > 0]
