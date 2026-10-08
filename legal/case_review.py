"""Source-preserving case/contract review workflow."""

from __future__ import annotations

import re

from .shared import retrieve, source_ref
from .contradiction import find_contradictions
from .missing_info import detect_case_missing_information


_PATTERNS = (
    (
        "date",
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})\b",
    ),
    (
        "amount",
        r"(?:₹|Rs\.?|INR|\$|USD)\s?[\d,]+(?:\.\d+)?",
    ),
    (
        "section",
        r"\b(?:Section|Sec\.?)\s*\d+[A-Za-z0-9()/-]*",
    ),
)


def _facts(hits: list[dict]) -> list[dict]:
    facts = []

    for hit in hits:
        found = False

        for kind, pattern in _PATTERNS:
            for match in re.finditer(pattern, hit["text"], re.IGNORECASE):
                facts.append(
                    {
                        "type": kind,
                        "value": match.group(0),
                        "source": source_ref(hit),
                    }
                )
                found = True

        if not found:
            facts.append(
                {
                    "type": "statement",
                    "value": hit["text"].strip(),
                    "source": source_ref(hit),
                }
            )

    return facts


def review_case(query: str, top_k: int = 5) -> dict:
    hits = retrieve(query, top_k)
    facts = _facts(hits)

    evidence = [
        {
            "statement": hit["text"],
            "source": source_ref(hit),
        }
        for hit in hits
    ]

    contradictions = find_contradictions(
        [
            {
                **source_ref(hit),
                "text": hit["text"],
            }
            for hit in hits
        ]
    )

    documents = []

    for hit in hits:
        document = next(
            (
                item
                for item in documents
                if item["document"] == hit["document"]
            ),
            None,
        )

        if document is None:
            document = {
                "document": hit["document"],
                "pages": [],
            }
            documents.append(document)

        if hit["page"] not in document["pages"]:
            document["pages"].append(hit["page"])

    return {
        "key_facts": facts,
        "evidence": evidence,
        "contradictions": contradictions,
        "missing_information": detect_case_missing_information(hits),
        "relevant_documents": documents,
    }
