from dataclasses import dataclass, field
from string import Formatter
from typing import Any, List, Tuple


@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-detect required variables from both templates
        formatter = Formatter()
        combined = self.system + self.user

        self.required_vars = [
            fname
            for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> Tuple[str, str]:
        """Return (rendered_system, rendered_user). Raises if vars are missing."""
        missing = set(self.required_vars) - set(kwargs)

        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        return (
            self.system.format(**kwargs),
            self.user.format(**kwargs),
        )


# =========================
# TEMPLATES
# =========================

QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="""You are a {domain} expert. Answer questions accurately and concisely.
Cite sources when possible. If you are unsure, say so.""",
    user="""Question: {question}

Context:
{context}""",
)

SUMMARY_TEMPLATE = PromptTemplate(
    name="document_summary",
    version="1.0",
    system="You are a technical writer. Summarise documents clearly for a {audience} audience.",
    user="""Summarise the following in {max_sentences} sentences or fewer:

{document}""",
)

# =========================
# USAGE
# =========================

system, user = QA_TEMPLATE.render(
    domain="machine learning",
    question="What is the vanishing gradient problem?",
    context="Gradients in deep networks are computed via backpropagation...",
)

print("System:\n", system)
print("\nUser:\n", user)
print("\nRequired vars:", QA_TEMPLATE.required_vars)