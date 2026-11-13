"""Stage 1b: field definitions, read from whatever the dataset provides.

A field's description comes from the schema when it has one (ExtractBench), otherwise from
the dataset's own definitions (CUAD: the "Details:" part of each question in CUADv1.json).
Nothing here names a specific field.
"""
from __future__ import annotations

from dataclasses import dataclass

from solution.data import cuad_definitions


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    kind: str  # scorer normaliser: "span", "string", "json", ...
    json_type: str

    @property
    def is_span(self) -> bool:
        return self.kind == "span"

    def query(self) -> str:
        """Text used for keyword and semantic retrieval."""
        return f"{self.name}. {self.description}".strip()


def field_specs(rec: dict) -> list[FieldSpec]:
    definitions = cuad_definitions() if rec.get("source_dataset") == "cuad" else {}
    specs = []
    for name, spec in (rec.get("schema") or {}).get("properties", {}).items():
        spec = spec if isinstance(spec, dict) else {}
        desc = spec.get("description") or definitions.get(name) or spec.get("title") or ""
        specs.append(FieldSpec(
            name=name,
            description=" ".join(str(desc).split())[:600],
            kind=spec.get("x-normalizer", "string"),
            json_type=str(spec.get("type", "string")),
        ))
    return specs
