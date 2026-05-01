import json

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from models import FunctionCall, FunctionDef
from constrained import (
    State,
    forbidden_ngram_ids,
    get_valid_function_prefixes,
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
        "You are a function-calling assistant. "
        "Given the user's request, select "
        "exactly one function from the list below and fill in its arguments. "
        "Respond with a single JSON object of the form "
        '{"function": "<name>", '
        '"arguments": {"<arg>": <value>, ...}} and nothing else.\n\n'
        "Guidelines for regex arguments:\n"
        "- To match all digits, ALWAYS use \\d+ (never list individual digits).\n"
        "- To match a set of characters, ALWAYS use [...] syntax: "
        "write [aeiouAEIOU] not aeiouAEIOU.\n"
        '- To match a literal word like cat, just write "cat".\n'
        "- Never repeat alternatives.\n\n"
        "Examples:\n"
        'Request: replace "cat" with "dog" in "the cat sat on the mat"\n'
        'Response: {"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "the cat sat on the mat", '
        '"regex": "cat", "replacement": "dog"}}\n'
        'Request: replace all vowels in "Programming is fun" with "*"\n'
        'Response: {"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "Programming is fun", "regex": "[aeiouAEIOU]", '
        '"replacement": "*"}}\n'
        'Request: replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS\n'
        'Response: {"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "Hello 34 I\'m 233 years old", '
        '"regex": "\\\\d+", "replacement": "NUMBERS"}}\n\n'
        f"Available functions:\n{functions_desc}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def generate(
    model: Small_LLM_Model,
    prompt_text: str,
    function_defs: list[FunctionDef],
    vocab: dict[int, str],
    target_function: str | None = None,
    max_tokens: int = 200,
) -> FunctionCall:
    """Generate a function call using constrained decoding."""
    valid_function_prefixes = get_valid_function_prefixes(function_defs)
    function_names = {fn.name for fn in function_defs}

    # If the function is already known (pass 2), find its definition upfront
    current_function: FunctionDef | None = None
    if target_function is not None:
        current_function = next(
            fn for fn in function_defs if fn.name == target_function
        )

    # Encode the prompt with function-calling context
    messages = _build_messages(prompt_text, function_defs)
    input_ids: list[int] = model.encode_chat(messages)[0].tolist()

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

    for _ in range(max_tokens):
        if state == State.END:
            break

        # Ask the model for the next token scores
        logits = model.get_logits_from_input_ids(input_ids)

        # Compute remaining fixed string for multi-token fixed states
        if state in _FIXED_STRINGS:
            remaining_fixed = _FIXED_STRINGS[state][len(accumulated_fixed):]
        else:
            remaining_fixed = ""

        # Keep only the tokens allowed by the current FSM state
        valid_ids = get_valid_tokens(
            state,
            vocab,
            valid_function_prefixes,
            current_function,
            remaining_params,
            accumulated_function_name,
            function_names,
            remaining_fixed,
            string_phase,
            accumulated_arg_key,
            value_token_count,
        )

        # No-repeat 3-gram on string arg values to avoid greedy decoding loops
        if state == State.ARG_VALUE and current_function and remaining_params:
            param_type = current_function.parameters[remaining_params[0]].type
            if param_type == "string":
                forbidden = forbidden_ngram_ids(value_token_ids, n=3)
                if forbidden:
                    filtered = [
                        tid for tid in valid_ids
                        if tid not in forbidden
                    ]
                    if filtered:
                        valid_ids = filtered

        logits = mask_logits(logits, valid_ids)

        # Greedy decoding: pick the token with the highest score
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

    return FunctionCall(
        prompt=prompt_text,
        name=data["function"],
        parameters=data["arguments"],
    )
