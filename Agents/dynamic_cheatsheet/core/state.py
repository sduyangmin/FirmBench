from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CheatsheetState:
    """
    Runtime state container for Dynamic Cheatsheet.
    """

    cheatsheet_text: str = "(empty)"
    previous_answers: List[str] = field(default_factory=list)
    input_history: List[str] = field(default_factory=list)
    output_history: List[str] = field(default_factory=list)
    embeddings: Optional[List[List[float]]] = None

    def update_cheatsheet(self, new_text: str) -> None:
        self.cheatsheet_text = new_text or self.cheatsheet_text

    def append_example(
        self,
        input_txt: str,
        generator_output: str,
        generator_answer: str,
    ) -> None:
        self.input_history.append(input_txt)
        self.output_history.append(generator_output)
        self.previous_answers.append(generator_answer)

    def snapshot(self) -> dict:
        return {
            "cheatsheet": self.cheatsheet_text,
            "previous_answers": list(self.previous_answers),
        }






















