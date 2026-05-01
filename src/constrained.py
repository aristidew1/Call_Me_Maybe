import json
from enum import Enum, auto
from typing import cast

import numpy as np

from models import FunctionDef


class State(Enum):
    """FSM states for constrained JSON decoding."""

    START = auto()
    OPEN_BRACE = auto()
    KEY_FUNCTION = auto()
    COLON = auto()
    FUNCTION_VALUE = auto()
    COMMA = auto()
    KEY_ARGUMENTS = auto()
    OPEN_ARGS = auto()
    ARG_KEY = auto()
    ARG_COLON = auto()
    ARG_VALUE = auto()
    ARG_COMMA = auto()
    CLOSE_ARGS = auto()
    CLOSE_BRACE = auto()
    END = auto()


def load_vocab(path: str) -> dict[int, str]:
    """Load the vocabulary file and return a mapping from token id
    to token string."""
    with open(path, "r") as file:
        vocab = json.load(file)
    return {token_id: token_str for token_str, token_id in vocab.items()}


def update_state(
    state: State,
    token: str,
    remaining_params: list[str],
    accumulated_fixed: str = "",
    accumulated_arg_key: str = "",
) -> State:
    """Return the next FSM state after generating a token."""
    match state:
        case State.START:
            return State.OPEN_BRACE
        case State.OPEN_BRACE:
            if accumulated_fixed == '"function"':
                return State.KEY_FUNCTION
            return State.OPEN_BRACE
        case State.KEY_FUNCTION:
            return State.COLON
        case State.COLON:
            return State.FUNCTION_VALUE
        case State.FUNCTION_VALUE:
            if token == '"':
                return State.COMMA
            return State.FUNCTION_VALUE
        case State.COMMA:
            return State.KEY_ARGUMENTS
        case State.KEY_ARGUMENTS:
            if accumulated_fixed == '"arguments":':
                return State.OPEN_ARGS
            return State.KEY_ARGUMENTS
        case State.OPEN_ARGS:
            return State.ARG_KEY if remaining_params else State.CLOSE_ARGS
        case State.ARG_KEY:
            target_key = f'"{remaining_params[0]}"'
            if remaining_params and accumulated_arg_key == target_key:
                return State.ARG_COLON
            return State.ARG_KEY
        case State.ARG_COLON:
            return State.ARG_VALUE
        case State.ARG_VALUE:
            if token == ',':
                return State.ARG_KEY
            if token == '}':
                return State.CLOSE_BRACE
            return State.ARG_VALUE
        case State.ARG_COMMA:
            return State.ARG_KEY
        case State.CLOSE_ARGS:
            return State.CLOSE_BRACE
        case State.CLOSE_BRACE:
            return State.END
        case _:
            return state


def _valid_ids_for_fixed_token(
    vocab: dict[int, str], expected: str
) -> list[int]:
    """Return token ids valid as a prefix of the expected fixed string."""
    return [tid for tid, tok in vocab.items() if expected.startswith(tok)]


def _valid_ids_for_number_value(
    vocab: dict[int, str], is_last_param: bool = False
) -> list[int]:
    """Return token ids valid inside a JSON number (digits, delimiters)."""
    number_chars = set("0123456789.-eE")
    delimiter = "}" if is_last_param else ","
    result = []
    for tid, tok in vocab.items():
        is_number_fragment = tok and all(c in number_chars for c in tok)
        if is_number_fragment or tok == delimiter:
            result.append(tid)
    return result


def _valid_ids_for_boolean_value(
    vocab: dict[int, str], is_last_param: bool = False
) -> list[int]:
    """Return token ids valid inside a JSON boolean (true/false prefixes)."""
    bool_prefixes = {
        "t", "tr", "tru", "true", "f", "fa", "fal", "fals", "false",
    }
    delimiter = "}" if is_last_param else ","
    result = []
    for tid, tok in vocab.items():
        is_bool_fragment = tok in bool_prefixes
        if is_bool_fragment or tok == delimiter:
            result.append(tid)
    return result


def _valid_ids_for_string_value(
    vocab: dict[int, str], string_phase: int = 0
) -> list[int]:
    """Return token ids valid for a JSON string value.

    string_phase: 0 = before opening quote, 1 = inside string,
    2 = after closing quote.
    """
    if string_phase == 0:
        return [tid for tid, tok in vocab.items() if tok == '"']
    if string_phase == 1:
        return [tid for tid, tok in vocab.items()
                if tok
                and all(32 <= ord(c) <= 126 for c in tok.replace('Ġ', ' '))
                and (tok == '"' or '"' not in tok)
                and "," not in tok and "}" not in tok and "{" not in tok]
    # phase 2: string is closed, only the correct delimiter allowed
    return []  # filled in caller with is_last_param context


def get_valid_tokens(
    state: State,
    vocab: dict[int, str],
    valid_function_prefixes: set[str],
    current_function: FunctionDef | None,
    remaining_params: list[str],
    accumulated_function_name: str = "",
    function_names: set[str] = set(),
    remaining_fixed: str = "",
    string_phase: int = 0,
    accumulated_arg_key: str = "",
    value_token_count: int = 0,
    max_value_tokens: int = 20,
) -> list[int]:
    """Return the list of valid token ids for the current FSM state."""
    match state:

        case State.START:
            return _valid_ids_for_fixed_token(vocab, "{")

        case State.OPEN_BRACE:
            target = remaining_fixed or '"function"'
            return _valid_ids_for_fixed_token(vocab, target)

        case State.KEY_FUNCTION:
            return _valid_ids_for_fixed_token(vocab, ":")

        case State.COLON:
            # Opening quote of the function name string
            return _valid_ids_for_fixed_token(vocab, '"')

        case State.FUNCTION_VALUE:
            result = []
            for tid, tok in vocab.items():
                candidate = accumulated_function_name + tok
                name_complete = (
                    tok == '"'
                    and accumulated_function_name in function_names
                )
                if candidate in valid_function_prefixes or name_complete:
                    result.append(tid)
            return result

        case State.COMMA:
            return _valid_ids_for_fixed_token(vocab, ",")

        case State.KEY_ARGUMENTS:
            target = remaining_fixed or '"arguments":'
            return _valid_ids_for_fixed_token(vocab, target)

        case State.OPEN_ARGS:
            return _valid_ids_for_fixed_token(vocab, "{")

        case State.ARG_KEY:
            if not current_function or not remaining_params:
                return []
            target = f'"{remaining_params[0]}"'
            remaining = target[len(accumulated_arg_key):]
            return [
                tid for tid, tok in vocab.items()
                if remaining.startswith(tok)
            ]

        case State.ARG_COLON:
            return _valid_ids_for_fixed_token(vocab, ":")

        case State.ARG_VALUE:
            if not current_function or not remaining_params:
                return []
            param_name = remaining_params[0]
            param_type = current_function.parameters[param_name].type
            is_last = len(remaining_params) == 1
            delimiter = "}" if is_last else ","
            # Force delimiter once enough value tokens have been generated —
            # but if we are still inside an open string, force the closing
            # quote first so the JSON stays valid.
            if value_token_count >= max_value_tokens:
                if param_type == "string" and string_phase == 1:
                    return [tid for tid, tok in vocab.items() if tok == '"']
                return [
                    tid for tid, tok in vocab.items()
                    if tok == delimiter
                ]
            if param_type == "number":
                return _valid_ids_for_number_value(vocab, is_last)
            if param_type == "boolean":
                return _valid_ids_for_boolean_value(vocab, is_last)
            if param_type == "string":
                if string_phase == 2:
                    return [
                        tid for tid, tok in vocab.items()
                        if tok == delimiter
                    ]
                return _valid_ids_for_string_value(vocab, string_phase)
            return []

        case State.ARG_COMMA:
            return _valid_ids_for_fixed_token(vocab, ",")

        case State.CLOSE_ARGS:
            return _valid_ids_for_fixed_token(vocab, "}")

        case State.CLOSE_BRACE:
            return _valid_ids_for_fixed_token(vocab, "}")

        case _:
            return []


def mask_logits(logits: list[float], valid_ids: list[int]) -> list[float]:
    """Set all logits to -inf except for valid token ids."""
    arr = np.array(logits, dtype=np.float32)
    mask = np.full(len(arr), False)
    mask[valid_ids] = True
    arr[~mask] = -np.inf
    return cast(list[float], arr.tolist())


def get_valid_function_prefixes(function_defs: list[FunctionDef]) -> set[str]:
    """Return all valid prefixes for all function names."""
    prefixes: set[str] = set()
    for fn in function_defs:
        for i in range(1, len(fn.name) + 1):
            prefixes.add(fn.name[0:i])
    return prefixes


def forbidden_ngram_ids(generated_ids: list[int], n: int = 3) -> set[int]:
    """Return token ids that would recreate an already-seen n-gram."""
    if len(generated_ids) < n - 1:
        return set()
    prefix = tuple(generated_ids[-(n - 1):])
    forbidden: set[int] = set()
    for i in range(len(generated_ids) - n + 1):
        ngram = tuple(generated_ids[i:i + n])
        if ngram[:-1] == prefix:
            forbidden.add(ngram[-1])
    return forbidden
