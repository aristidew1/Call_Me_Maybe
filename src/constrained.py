import json
from dataclasses import dataclass
from enum import Enum, auto

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
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return {token_id: token_str for token_str, token_id in vocab.items()}


@dataclass
class TokenSets:
    """Token ID sets pre-computed once from the vocabulary.

    Instead of scanning all tokens on every generation step,
    each set is built once and looked up in O(1) during decoding.
    """

    # Maps a target string (or suffix of a fixed string) to the token IDs
    # that match it — i.e. tokens `tok` where `target.startswith(tok)`.
    # Covers every fixed string used in the FSM (single-char and multi-char).
    fixed: dict[str, list[int]]

    # Number value tokens, split by which delimiter comes after the value
    number_comma: list[int]   # mid-argument: delimiter is ","
    number_brace: list[int]   # last argument: delimiter is "}"

    # Boolean value tokens, same split
    bool_comma: list[int]
    bool_brace: list[int]

    # String value tokens for the "inside string" phase (printable chars + '"')
    string_inside: list[int]
    # Single-character variant of string_inside, used for short-budget params
    # (e.g. `replacement`) to avoid multi-char BPE tokens like "*uc".
    string_inside_single: list[int]

    # Single-delimiter sets, used to force-close a value when max tokens reached
    delimiter_comma: list[int]
    delimiter_brace: list[int]
    closing_quote: list[int]

    # Maps each possible accumulated function-name prefix to the tokens that
    # can legally follow it (extends a valid prefix or closes with '"').
    function_value: dict[str, list[int]]


def build_token_sets(
    vocab: dict[int, str],
    function_defs: list[FunctionDef],
) -> TokenSets:
    """Pre-compute every token set needed during constrained decoding.

    Call this once before the generation loop; pass the result to
    get_valid_tokens instead of the raw vocabulary.
    """

    # --- Fixed-string sets ------------------------------------------------
    # Every target string (and each suffix of multi-token targets) that
    # get_valid_tokens will ever need to look up.
    fixed_targets: set[str] = set()

    for s in ["{", ":", '"', ",", "}"]:
        fixed_targets.add(s)

    for full_str in ['"function"', '"arguments":']:
        for i in range(len(full_str)):
            fixed_targets.add(full_str[i:])

    for fn in function_defs:
        for param_name in fn.parameters:
            target = f'"{param_name}"'
            for i in range(len(target)):
                fixed_targets.add(target[i:])

    fixed = {
        target: [tid for tid, tok in vocab.items() if target.startswith(tok)]
        for target in fixed_targets
    }

    # --- Delimiter / quote sets -------------------------------------------
    comma_tids  = [tid for tid, tok in vocab.items() if tok == ","]
    brace_tids  = [tid for tid, tok in vocab.items() if tok == "}"]
    quote_tids  = [tid for tid, tok in vocab.items() if tok == '"']

    # --- Number value sets ------------------------------------------------
    number_chars = set("0123456789.-eE")
    number_tids = [
        tid for tid, tok in vocab.items()
        if tok and all(c in number_chars for c in tok)
    ]
    number_comma = number_tids + comma_tids
    number_brace = number_tids + brace_tids

    # --- Boolean value sets -----------------------------------------------
    bool_prefixes = {"t", "tr", "tru", "true", "f", "fa", "fal", "fals", "false"}
    bool_tids = [tid for tid, tok in vocab.items() if tok in bool_prefixes]
    bool_comma = bool_tids + comma_tids
    bool_brace = bool_tids + brace_tids

    # --- String (inside) set ----------------------------------------------
    string_inside = [
        tid for tid, tok in vocab.items()
        if tok
        and all(32 <= ord(c) <= 126 for c in tok.replace("Ġ", " "))
        and (tok == '"' or '"' not in tok)
    ]
    # Single-character-only variant for the `replacement` param: forces
    # char-by-char generation to avoid multi-char BPE tokens like "*uc"
    # that hijack the greedy decoder. Excludes "\" so the model cannot emit
    # invalid JSON escape sequences like "\1".
    string_inside_single = [
        tid for tid, tok in vocab.items()
        if tok and len(tok.replace("Ġ", " ")) == 1
        and 32 <= ord(tok.replace("Ġ", " ")) <= 126
        and tok not in {'"', "\\"}
    ]

    # --- Function-value sets ----------------------------------------------
    function_names = {fn.name for fn in function_defs}
    valid_function_prefixes = get_valid_function_prefixes(function_defs)

    function_value: dict[str, list[int]] = {}
    for fn in function_defs:
        for i in range(len(fn.name) + 1):       # "" through the full name
            accumulated = fn.name[:i]
            if accumulated in function_value:
                continue
            result = []
            for tid, tok in vocab.items():
                candidate = accumulated + tok
                name_complete = (tok == '"' and accumulated in function_names)
                if candidate in valid_function_prefixes or name_complete:
                    result.append(tid)
            function_value[accumulated] = result

    return TokenSets(
        fixed=fixed,
        number_comma=number_comma,
        number_brace=number_brace,
        bool_comma=bool_comma,
        bool_brace=bool_brace,
        string_inside=string_inside,
        string_inside_single=string_inside_single,
        delimiter_comma=comma_tids,
        delimiter_brace=brace_tids,
        closing_quote=quote_tids,
        function_value=function_value,
    )


def update_state(
    state: State,
    token: str,
    remaining_params: list[str],
    accumulated_fixed: str = "",
    accumulated_arg_key: str = "",
    string_phase: int = 0,
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
            # Don't transition on , or } while inside an open string literal
            if string_phase == 1:
                return State.ARG_VALUE
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


_REGEX_TERMINAL_CHARS = {"+", "*", "?", "]", ")", "$"}


def is_regex_complete(
    value_token_ids: list[int], vocab: dict[int, str]
) -> bool:
    """Return True if the last generated character is a regex-terminal char.

    After producing `+`, `*`, `?`, `]`, `)` the regex pattern can be
    considered complete — forcing the closing quote here prevents the
    greedy decoder from extending a valid pattern (e.g. `\\d+` → `\\d+\\d+$`).
    """
    if not value_token_ids:
        return False
    last_tok = vocab.get(value_token_ids[-1], "")
    if not last_tok:
        return False
    last_char = last_tok.replace("Ġ", " ")[-1]
    return last_char in _REGEX_TERMINAL_CHARS


def get_valid_tokens(
    state: State,
    token_sets: TokenSets,
    current_function: FunctionDef | None,
    remaining_params: list[str],
    accumulated_function_name: str = "",
    remaining_fixed: str = "",
    string_phase: int = 0,
    accumulated_arg_key: str = "",
    value_token_count: int = 0,
    max_value_tokens: int = 20,
    value_token_ids: list[int] | None = None,
    vocab: dict[int, str] | None = None,
) -> list[int]:
    """Return the list of valid token ids for the current FSM state.

    All sets are O(1) lookups into token_sets — no vocabulary scanning here.
    """
    match state:

        case State.START:
            return token_sets.fixed["{"]

        case State.OPEN_BRACE:
            target = remaining_fixed or '"function"'
            return token_sets.fixed[target]

        case State.KEY_FUNCTION:
            return token_sets.fixed[":"]

        case State.COLON:
            return token_sets.fixed['"']

        case State.FUNCTION_VALUE:
            return token_sets.function_value.get(accumulated_function_name, [])

        case State.COMMA:
            return token_sets.fixed[","]

        case State.KEY_ARGUMENTS:
            target = remaining_fixed or '"arguments":'
            return token_sets.fixed[target]

        case State.OPEN_ARGS:
            return token_sets.fixed["{"]

        case State.ARG_KEY:
            if not current_function or not remaining_params:
                return []
            target = f'"{remaining_params[0]}"'
            remaining = target[len(accumulated_arg_key):]
            return token_sets.fixed.get(remaining, [])

        case State.ARG_COLON:
            return token_sets.fixed[":"]

        case State.ARG_VALUE:
            if not current_function or not remaining_params:
                return []
            param_type = current_function.parameters[remaining_params[0]].type
            is_last = len(remaining_params) == 1

            if value_token_count >= max_value_tokens:
                if param_type == "string" and string_phase == 1:
                    return token_sets.closing_quote
                return token_sets.delimiter_brace if is_last else token_sets.delimiter_comma

            if param_type in ("number", "integer"):
                return token_sets.number_brace if is_last else token_sets.number_comma
            if param_type == "boolean":
                return token_sets.bool_brace if is_last else token_sets.bool_comma
            if param_type == "string":
                if string_phase == 0:
                    return token_sets.fixed['"']
                if string_phase == 1:
                    param_name = remaining_params[0]
                    # Regex-completion heuristic: after a terminal regex char
                    # (`+`, `*`, `?`, `]`, `)`), force closing quote so the
                    # greedy decoder cannot extend a complete pattern.
                    if (
                        param_name == "regex"
                        and value_token_ids
                        and vocab is not None
                        and is_regex_complete(value_token_ids, vocab)
                    ):
                        return token_sets.closing_quote
                    # Forbid leading whitespace right after the opening quote
                    if value_token_count == 1 and vocab is not None:
                        return [
                            tid for tid in token_sets.string_inside
                            if not vocab[tid].startswith(("Ġ", " "))
                        ]
                    return token_sets.string_inside
                # phase 2: string is closed, only the delimiter is valid
                return token_sets.delimiter_brace if is_last else token_sets.delimiter_comma
            return []

        case State.ARG_COMMA:
            return token_sets.fixed[","]

        case State.CLOSE_ARGS:
            return token_sets.fixed["}"]

        case State.CLOSE_BRACE:
            return token_sets.fixed["}"]

        case _:
            return []


def mask_logits(logits: list[float], valid_ids: list[int]) -> np.ndarray:
    """Set all logits to -inf except for valid token ids.

    Returns a numpy array directly so the caller can pass it straight to
    np.argmax without an extra list conversion.
    """
    arr = np.array(logits, dtype=np.float32)
    mask = np.full(len(arr), False)
    mask[valid_ids] = True
    arr[~mask] = -np.inf
    return arr


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
