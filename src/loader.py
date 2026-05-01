import json
import sys

from models import FunctionDef, Prompt


def load_function_definitions(path: str) -> list[FunctionDef]:
    """Load and validate function definitions from a JSON file."""
    functions: list[FunctionDef] = []
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(
            f"Error: functions definition file not found at '{path}'",
            file=sys.stderr,
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error: invalid JSON in '{path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    for item in data:
        functions.append(FunctionDef(**item))
    return functions


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate prompts from a JSON file."""
    prompts: list[Prompt] = []
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: prompts file not found at '{path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in '{path}': {e}")
        sys.exit(1)
    for item in data:
        prompts.append(Prompt(**item))
    return prompts
