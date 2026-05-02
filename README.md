# Call Me Maybe

Translate natural-language prompts into structured function calls using `Qwen/Qwen3-0.6B` and **constrained decoding** built from scratch (no `outlines`, `transformers`, `dspy`, etc.).

```
"What is the sum of 40 and 2?"
        │
        ▼
{"function": "add_numbers", "arguments": {"a": 40, "b": 2}}
```

---

## Description

Small language models are unreliable at producing structured JSON when prompted naïvely (~30% success). This project reaches near-perfect JSON validity by **masking the logits at every generation step**: at each token, an FSM tracks where we are in the JSON skeleton and only the token IDs that keep the output valid are kept; everything else is set to `-inf` before `argmax`.

The function name is chosen **by the LLM** (constrained to the set of valid function names), the arguments are extracted by the LLM (constrained to the right type — number / string / boolean — for each parameter).

---

## Installation

```bash
make install     # uv sync
```

Requires Python 3.10+. Dependencies: `numpy`, `pydantic`, `llm-sdk` (local editable).

## Usage

```bash
uv run python -m src \
  [--functions_definition data/input/functions_definition.json] \
  [--input data/input/function_calling_tests.json] \
  [--output data/output/function_calls.json]
```

Or via the Makefile:

```bash
make run                                  # default paths
make run ARGS="--input my_prompts.json"   # custom input
make debug                                # run with -Wall
make clean                                # purge caches
make lint                                 # flake8 + mypy
```

### Output schema

Each entry of the output JSON contains exactly:

```json
{
  "prompt":     "What is the sum of 40 and 2?",
  "name":       "add_numbers",
  "parameters": {"a": 40, "b": 2}
}
```

---

## Algorithm

### 1. FSM-driven JSON skeleton

A finite-state machine ([src/constrained.py](src/constrained.py), `class State`) tracks the current position in the target JSON:

```
START → OPEN_BRACE → KEY_FUNCTION → COLON → FUNCTION_VALUE
      → COMMA → KEY_ARGUMENTS → OPEN_ARGS
      → [ARG_KEY → ARG_COLON → ARG_VALUE]*
      → CLOSE_ARGS → CLOSE_BRACE → END
```

At every step `get_valid_tokens(state, …)` returns the list of token IDs that can legally extend the partial JSON.

### 2. Logit masking

```python
logits = model.get_logits_from_input_ids(input_ids)
valid  = get_valid_tokens(state, …)
logits = mask_logits(logits, valid)        # invalid → -inf
next_token = int(np.argmax(logits))        # greedy decode
```

This is the only place the model is constrained — there is no post-hoc parsing or retry: every token generated is, by construction, syntactically valid.

### 3. Type-aware value generation

Inside `ARG_VALUE`, the FSM uses the parameter's type from the Pydantic `FunctionDef`:

| type      | allowed tokens                                 |
| --------- | ---------------------------------------------- |
| `number`  | digits, `.`, `-`, `e`, plus the next delimiter |
| `boolean` | prefixes of `true` / `false` + delimiter       |
| `string`  | printable ASCII inside `"…"`, then delimiter   |

A 3-gram no-repeat filter is applied on string values to prevent greedy-decoding loops, and a regex-completion heuristic forces the closing quote after a terminal regex char (`+ * ? ] )`) so the decoder can't extend an already-valid pattern.

### 4. Function-name lookup

All valid prefixes of every function name are pre-computed once in `build_token_sets`, so each step is an O(1) lookup instead of a vocabulary scan.

---

## Project structure

```
Call_Me_Maybe/
├── src/
│   ├── __main__.py        # CLI entry point (argparse)
│   ├── models.py          # Pydantic models (FunctionDef, Prompt, FunctionCall)
│   ├── loader.py          # JSON input loading + validation
│   ├── constrained.py     # FSM + token-set pre-computation + logit masking
│   ├── generator.py       # Token-by-token generation loop
│   └── writer.py          # JSON output serialization
├── llm_sdk/               # Provided LLM SDK (editable install)
├── data/
│   ├── input/             # functions_definition.json + function_calling_tests.json
│   └── output/            # generated at runtime
├── Makefile
├── pyproject.toml
└── uv.lock
```

---

## Design decisions

- **Pure FSM, no parser combinators.** The output JSON has a fixed shape, so a hand-written FSM with ~15 states is simpler and faster than a generic JSON-grammar engine.
- **Pre-computed token sets.** Every set the FSM might need (delimiters, number tokens, boolean prefixes, string-inside chars, every prefix of every function name) is built once in `build_token_sets` and looked up by reference. No vocabulary scan per token.
- **Single greedy pass, no retry.** Because masking guarantees validity, there is no JSON-repair fallback — `json.loads` on the final output always succeeds.
- **Public SDK methods only.** The pipeline uses `encode_chat`, `get_logits_from_input_ids`, `decode`, `get_path_to_vocab_file`, and `reset_kv_cache`. No private attributes are accessed.
- **KV-cache reuse.** `model.reset_kv_cache()` is called once per prompt; within a generation the SDK reuses cached attention states.

---

## Challenges encountered

- **Multi-character BPE tokens hijacking string values.** The greedy decoder would emit tokens like `*uc` mid-string for short replacement values. Fixed by a single-character variant of the string-inside set when the parameter has a tight token budget.
- **Greedy regex extension.** A valid pattern like `\d+` would be extended into `\d+\d+$` by the greedy argmax. Fixed by `is_regex_complete`: after a terminal regex char (`+ * ? ] )`) the FSM forces the closing quote.
- **Function-name multi-token resolution.** Token boundaries don't align with function names. Fixed by pre-computing, for every prefix of every name, the tokens that legally extend it (or close with `"`).
- **String repetition loops.** Greedy decoding on long string args occasionally looped. Fixed with a 3-gram no-repeat filter on `value_token_ids`.

---

## Performance

- 100% of outputs are valid JSON (guaranteed by construction).
- ~90% accuracy on the moulinette `private` set.
- All test prompts processed well under 5 minutes on CPU.

Optimizations:

- Pre-computed token sets (one-time O(|V|) scan at startup, O(1) per step).
- KV-cache reuse via `model.reset_kv_cache()` between prompts.
- Indexed function-name prefixes (`function_value: dict[str, list[int]]`) — no scan during decoding.
- Greedy decode with FSM masking, single forward pass per token, no retry / repair phase.

---

## Testing

```bash
# End-to-end on the provided test set
make run

# Moulinette evaluation (from the moulinette/ folder)
cd moulinette
uv sync
uv run python -m moulinette prepare_exercises --set private
uv run python -m moulinette grade_student_answers \
    --student_answer_path ../data/output/function_calls.json --set private
```

The grader checks: prompt match, function-name validity, argument types, and the actual function output against the expected output.

---

## Resources

- Subject: [subject.md](subject.md)
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Constrained decoding background: the approach used here (FSM + logit masking) is a hand-rolled version of what libraries like `outlines` or `lm-format-enforcer` do under the hood.

---

## AI usage

AI assistants (Claude) were used during development for:

- Brainstorming the FSM state layout and edge-case handling (regex completion, BPE hijacking).
- Drafting docstrings and inline comments.
- Reviewing the constrained-decoding loop for correctness.

All generated suggestions were read, understood, tested, and adapted before being kept. No AI-generated code was committed without manual verification.
