"""Write function call results to a JSON file."""

import json
from pathlib import Path
from models import FunctionCall


def write_results(
    results: list[FunctionCall], output_path: Path | str
) -> None:
    """Serialize a list of FunctionCall to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
