"""HotpotQA (distractor config): questions with sentence-level gold evidence, plus the 2017 opening
paragraph of every Wikipedia article the questions reference."""

from dataclasses import dataclass

HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"


@dataclass(frozen=True)
class Question:
    id: str
    split: str  # "train" or "validation"
    question: str
    answer: str
    type: str  # "bridge" or "comparison"
    level: str  # "easy" / "medium" (train only) or "hard"
    supporting: tuple[tuple[str, int], ...]  # (title, sentence index) gold evidence
    context_titles: tuple[str, ...]  # 2 gold + 8 distractor articles

    @property
    def gold_titles(self) -> frozenset[str]:
        return frozenset(title for title, _ in self.supporting)


def load(splits: tuple[str, ...] = ("train", "validation")):
    """Returns (questions, paragraphs) where paragraphs maps title -> list of 2017 sentences."""
    from datasets import load_dataset

    questions: list[Question] = []
    paragraphs: dict[str, list[str]] = {}
    for split in splits:
        for row in load_dataset(HOTPOT_REPO, HOTPOT_CONFIG, split=split):
            ctx = row["context"]
            for title, sentences in zip(ctx["title"], ctx["sentences"]):
                # The same article appears in many questions with the same text; keep the first.
                paragraphs.setdefault(title, sentences)
            sf = row["supporting_facts"]
            questions.append(
                Question(
                    id=row["id"],
                    split=split,
                    question=row["question"],
                    answer=row["answer"],
                    type=row["type"],
                    level=row["level"],
                    supporting=tuple(zip(sf["title"], sf["sent_id"])),
                    context_titles=tuple(ctx["title"]),
                )
            )
    return questions, paragraphs
