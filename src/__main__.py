"""CLI entry point for Call Me Maybe function-calling pipeline."""

import argparse
from constrained import build_token_sets, load_vocab
from pathlib import Path
from loader import load_function_definitions, load_prompts
from generator import generate
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from writer import write_results
from models import FunctionCall


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts "
            "into structured function calls."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing prompts to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("data/output").mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.input)
    functions = load_function_definitions(args.functions_definition)
    model = Small_LLM_Model()
    vocab = load_vocab(model.get_path_to_vocab_file())
    token_sets = build_token_sets(vocab, functions)
    functions_call: list[FunctionCall] = []
    for p in prompts:
        functions_call.append(generate(
            prompt_text=p.prompt,
            function_defs=functions,
            model=model,
            vocab=vocab,
            token_sets=token_sets,
        ))
    write_results(results=functions_call, output_path=args.output)
