import json
import re

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from models import FunctionCall, FunctionDef
from constrained import (
    State,
    TokenSets,
    build_token_sets,
    forbidden_ngram_ids,
    get_valid_tokens,
    mask_logits,
    update_state,
)


def _build_messages(
    user_prompt: str, function_defs: list[FunctionDef]
) -> list[dict]:
    """Build chat messages that give the model the function-calling context."""
    lines = []
    for fn in function_defs:
        params = ", ".join(
            f"{pn}: {pd.type}" for pn, pd in fn.parameters.items()
        )
        desc = f"- {fn.name}({params}) -> {fn.returns.type}: {fn.description}"
        lines.append(desc)
    functions_desc = "\n".join(lines)
    system = (
        "Pick one function and fill its arguments. "
        "Regex tips: match any digit with \\\\d+ (double backslash for JSON), "
        "match a char set with [abc], literal text is just written as-is.\n"
        f"Functions:\n{functions_desc}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten chat messages into a single prompt string."""
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def generate(
    model: Small_LLM_Model,
    prompt_text: str,
    function_defs: list[FunctionDef],
    vocab: dict[int, str],
    token_sets: TokenSets | None = None,
    target_function: str | None = None,
    max_tokens: int = 200,
) -> FunctionCall:
    """Generate a function call using constrained decoding."""
    if token_sets is None:
        token_sets = build_token_sets(vocab, function_defs)

    # New prompt → fresh KV cache
    model.reset_kv_cache()

    # If the function is already known (pass 2), find its definition upfront
    current_function: FunctionDef | None = None
    if target_function is not None:
        current_function = next(
            fn for fn in function_defs if fn.name == target_function
        )

    # Encode the prompt with function-calling context
    messages = _build_messages(prompt_text, function_defs)
    prompt = _messages_to_prompt(messages)
    input_ids: list[int] = model.encode(prompt)[0].tolist()

    # Track FSM state and generated JSON
    state = State.START
    remaining_params: list[str] = []
    accumulated_function_name = ""
    accumulated_fixed = ""
    accumulated_arg_key = ""
    string_phase = 0
    value_token_count = 0
    generated_ids: list[int] = []
    value_token_ids: list[int] = []

    _FIXED_STRINGS = {
        State.OPEN_BRACE: '"function"',
        State.KEY_ARGUMENTS: '"arguments":',
    }

    # States where output is fully determined by the FSM, so we can pick the
    # longest valid token directly and skip the (expensive) model forward pass.
    _DETERMINISTIC_STATES = {
        State.START, State.OPEN_BRACE, State.KEY_FUNCTION, State.COLON,
        State.COMMA, State.KEY_ARGUMENTS, State.OPEN_ARGS, State.ARG_KEY,
        State.ARG_COLON, State.ARG_COMMA, State.CLOSE_ARGS, State.CLOSE_BRACE,
    }

    for _ in range(max_tokens):
        if state == State.END:
            break

        # Compute remaining fixed string for multi-token fixed states
        if state in _FIXED_STRINGS:
            remaining_fixed = _FIXED_STRINGS[state][len(accumulated_fixed):]
        else:
            remaining_fixed = ""

        # Keep only the tokens allowed by the current FSM state
        max_value_tokens = (
            current_function.parameters[remaining_params[0]].max_tokens
            if current_function and remaining_params
            else 20
        )
        valid_ids = get_valid_tokens(
            state,
            token_sets,
            current_function,
            remaining_params,
            accumulated_function_name,
            remaining_fixed,
            string_phase,
            accumulated_arg_key,
            value_token_count,
            max_value_tokens,
            value_token_ids,
            vocab,
        )

        if state in _DETERMINISTIC_STATES and valid_ids:
            # Skip the model: pick the longest valid token to advance fastest.
            next_token_id = max(valid_ids, key=lambda tid: len(vocab[tid]))
        else:
            logits = model.get_logits_incremental(input_ids)

            # No-repeat 3-gram on string values to avoid greedy decoding loops
            if (
                state == State.ARG_VALUE
                and current_function and remaining_params
                and current_function.parameters[remaining_params[0]].type == "string"
            ):
                forbidden = forbidden_ngram_ids(value_token_ids, n=3)
                if forbidden:
                    filtered = [tid for tid in valid_ids if tid not in forbidden]
                    if filtered:
                        valid_ids = filtered

            logits = mask_logits(logits, valid_ids)
            next_token_id = int(np.argmax(logits))

        next_token_str = vocab[next_token_id]

        input_ids.append(next_token_id)
        generated_ids.append(next_token_id)
        if state == State.ARG_VALUE:
            value_token_ids.append(next_token_id)
        print(next_token_str, end="", flush=True)

        # Accumulate fixed strings for multi-token states
        if state in _FIXED_STRINGS:
            accumulated_fixed += next_token_str

        # Accumulate the function name while inside FUNCTION_VALUE
        if state == State.FUNCTION_VALUE and next_token_str != '"':
            accumulated_function_name += next_token_str

        # Accumulate arg key
        if state == State.ARG_KEY:
            accumulated_arg_key += next_token_str

        # Count tokens generated for the current argument value
        if state == State.ARG_VALUE:
            value_token_count += 1

        # Track string phase: 0=before opening ", 1=inside, 2=after closing "
        if state == State.ARG_VALUE and current_function and remaining_params:
            param = current_function.parameters[remaining_params[0]]
            if param.type == "string":
                if next_token_str == '"' and string_phase in {0, 1}:
                    string_phase += 1

        # Transition to the next FSM state
        previous_state = state
        state = update_state(
            state, next_token_str, remaining_params,
            accumulated_fixed, accumulated_arg_key,
        )

        # Reset fixed accumulator on state change
        if state != previous_state:
            accumulated_fixed = ""

        # Reset string phase and value counter when leaving ARG_VALUE
        if previous_state == State.ARG_VALUE and state != State.ARG_VALUE:
            string_phase = 0
            value_token_count = 0
            value_token_ids = []

        # Reset arg key accumulator when leaving ARG_KEY
        if previous_state == State.ARG_KEY and state != State.ARG_KEY:
            accumulated_arg_key = ""

        # Function name complete: resolve to FunctionDef
        if previous_state == State.FUNCTION_VALUE and state == State.COMMA:
            current_function = next(
                fn for fn in function_defs
                if fn.name == accumulated_function_name
            )
            remaining_params = list(current_function.parameters.keys())

        # Once an argument value is complete, remove it from the remaining list
        done_states = {State.ARG_KEY, State.CLOSE_ARGS, State.CLOSE_BRACE}
        if previous_state == State.ARG_VALUE and state in done_states:
            remaining_params.pop(0)

    # Parse the full generated JSON and build a FunctionCall
    print()
    generated_json = model.decode(generated_ids)
    data = json.loads(generated_json)

    arguments = _post_process_arguments(data["arguments"], prompt_text)

    return FunctionCall(
        prompt=prompt_text,
        name=data["function"],
        parameters=arguments,
    )


_WORD_PROMPT_RE = re.compile(
    r"\bword\s+['\"]([^'\"\n]+)['\"]",
    re.IGNORECASE,
)


def _post_process_arguments(args: dict, prompt_text: str) -> dict:
    """Heuristic fixes the greedy decoder cannot get right on its own.

    1. ``replacement``: collapse "**" / "---" / "===" (single char repeated)
       to one char. Greedy BPE often picks these as a multi-char token even
       when the user asked for an "asterisk" / "dash" / etc.
    2. ``regex``: when the user prompt mentions a quoted word ("substitute
       the word 'cat'"), force word-boundary anchoring `\\bX\\b` so the
       pattern doesn't match substrings.
    """
    fixed = dict(args)

    repl = fixed.get("replacement")
    if isinstance(repl, str) and len(repl) >= 2 and len(set(repl)) == 1:
        fixed["replacement"] = repl[0]

    rgx = fixed.get("regex")
    if isinstance(rgx, str):
        match = _WORD_PROMPT_RE.search(prompt_text)
        if match:
            target = re.escape(match.group(1))
            if not rgx.startswith("\\b"):
                fixed["regex"] = f"\\b{target}\\b"

    return fixed
