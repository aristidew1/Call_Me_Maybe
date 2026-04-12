# Plan complet — Call Me Maybe

## Objectif du projet

Créer un outil de **function calling** qui traduit des prompts en langage naturel en appels de fonctions JSON structurés, en utilisant le modèle `Qwen/Qwen3-0.6B` avec du **constrained decoding** fait maison (sans outlines, transformers, dspy, etc.).

---

## Architecture des fichiers

```
Call_Me_Maybe/
├── src/
│   ├── __init__.py          # Point d'entrée (-m src)
│   ├── __main__.py          # CLI argparse
│   ├── models.py            # Pydantic models (FunctionDef, Prompt, Result)
│   ├── loader.py            # Chargement/validation des JSON d'entrée
│   ├── constrained.py       # Moteur de constrained decoding
│   ├── generator.py         # Pipeline génération token-by-token
│   └── writer.py            # Écriture du JSON de sortie
├── llm_sdk/                 # SDK fourni (copie)
│   └── __init__.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/              # Généré à l'exécution (ignoré git)
├── pyproject.toml           # uv, dépendances: numpy, pydantic
├── uv.lock
├── Makefile
├── README.md
└── .gitignore
```

---

## Étapes d'implémentation

### 1. Setup projet (`pyproject.toml`, `Makefile`, `.gitignore`)

- `pyproject.toml` : Python 3.10+, deps = `numpy`, `pydantic`
- `Makefile` : règles `install`, `run`, `debug`, `clean`, `lint`, `lint-strict`
- `.gitignore` : `__pycache__`, `.mypy_cache`, `.venv`, `data/output/`

### 2. Modèles Pydantic (`src/models.py`)

```python
class ParameterDef(BaseModel)    # {"type": "number"|"string"|"boolean"}
class FunctionDef(BaseModel)     # name, description, parameters, returns
class Prompt(BaseModel)          # {"prompt": str}
class FunctionCall(BaseModel)    # prompt, name, parameters
```

### 3. Chargement des inputs (`src/loader.py`)

- Lit `functions_definition.json` → `List[FunctionDef]`
- Lit `function_calling_tests.json` → `List[Prompt]`
- Gestion d'erreur : fichier manquant, JSON invalide → message clair, pas de crash

### 4. Moteur de constrained decoding (`src/constrained.py`)

C'est le **cœur du projet**. Approche : **JSON Schema guided token masking**.

#### Principe

À chaque étape de génération, on maintient un **état de parsing** qui décrit où on en est dans le JSON attendu. On calcule l'ensemble des tokens valides, on met les autres à `-inf`.

#### Schema de sortie fixe

Le JSON à générer pour chaque prompt est :
```json
{"function": "<nom>", "arguments": {<args>}}
```

#### États du parser (FSM)

```
START → OPEN_BRACE → KEY_FUNCTION → COLON → FUNCTION_VALUE
      → COMMA → KEY_ARGUMENTS → COLON → OPEN_ARGS
      → [pour chaque param: KEY → COLON → VALUE → COMMA/CLOSE]
      → CLOSE_ARGS → CLOSE_BRACE → END
```

#### Implémentation clé

1. **Charger le vocabulaire** via `llm_sdk.get_path_to_vocabulary_json()` → dict `{token_id: token_str}`
2. Pour chaque état, calculer les **token_ids valides** :
   - État "FUNCTION_VALUE" : seuls les token_ids dont le token_str correspond à un nom de fonction valide
   - État "NUMBER_VALUE" : tokens dont la concaténation reste un nombre valide (chiffres, `.`, `-`, `e`)
   - État "STRING_VALUE" : tout token sauf les caractères de contrôle non échappés
   - État "BOOLEAN_VALUE" : tokens de `true`/`false`
3. Masquer les logits invalides → `-inf` (via numpy)
4. Sélectionner le token avec le score max (greedy)

#### Stratégie en 2 passes

**Passe 1 — Sélection de fonction** (avec constrained decoding) :
- Prompt : `"Given these functions: [...descriptions...]\nUser: <prompt>\nCall:"`
- Générer uniquement le champ `"function"` → contraint aux noms valides
- Le LLM choisit la bonne fonction par compréhension sémantique

**Passe 2 — Extraction des arguments** :
- Une fois la fonction connue, on connaît le schema exact des params
- Générer les arguments en contraignant chaque valeur au bon type

### 5. Pipeline de génération (`src/generator.py`)

```python
def generate(
    model: Small_LLM_Model,
    prompt_text: str,
    function_defs: List[FunctionDef],
    vocab: Dict[int, str],
    target_function: Optional[str] = None,
    max_tokens: int = 200,
) -> FunctionCall
```

Boucle token-by-token :
```
input_ids = encode(prompt)
while not finished:
    logits = get_logits_from_input_ids(input_ids)
    valid_mask = compute_valid_tokens(state, partial_json, vocab)
    logits[~valid_mask] = -inf
    next_token = argmax(logits)
    input_ids.append(next_token)
    state = update_state(state, vocab[next_token])
```

### 6. CLI (`src/__main__.py`)

```
uv run python -m src \
  [--functions_definition data/input/functions_definition.json] \
  [--input data/input/function_calling_tests.json] \
  [--output data/output/function_calls.json]
```

- Argparse avec valeurs par défaut
- Crée `data/output/` si nécessaire
- Affiche progression (nombre de prompts traités)

### 7. Écriture des résultats (`src/writer.py`)

- Sérialise `List[FunctionCall]` → JSON valide
- Types Python → types JSON corrects (`float` pour number, `str` pour string, `bool` pour boolean)

---

## Points critiques à maîtriser

| Point | Difficulté | Solution |
|---|---|---|
| Token partiel vs complet | Haute | Tester si `current_json + token` reste un préfixe valide du JSON attendu |
| Nombres multi-tokens | Haute | FSM numérique : accumuler jusqu'au délimiteur suivant |
| Strings multi-tokens | Moyenne | Accumuler jusqu'au `"` fermant non échappé |
| Noms de fonctions multi-tokens | Haute | Pré-calculer tous les préfixes valides de chaque nom |
| Paramètres typés correctement | Moyenne | Convertir la string générée vers le type Pydantic attendu |

---

## Contraintes techniques rappelées

- Python 3.10+, flake8, mypy (type hints partout, docstrings PEP 257)
- Pydantic pour toutes les classes
- `numpy` et `json` autorisés
- **Interdit** : dspy, pytorch, transformers, huggingface, outlines
- Modèle : `Qwen/Qwen3-0.6B` (via `llm_sdk`)
- La **fonction doit être choisie par le LLM**, pas par heuristique
- `uv sync` suffit pour installer

---

## Ordre de développement recommandé

1. `pyproject.toml` + `Makefile` + `.gitignore`
2. `src/models.py` — Pydantic models
3. `src/loader.py` — I/O fichiers
4. Créer les fichiers `data/input/` de test
5. `src/constrained.py` — FSM + masquage de logits (étape la plus longue)
6. `src/generator.py` — pipeline complet
7. `src/__main__.py` — CLI
8. `src/writer.py` — sérialisation
9. Tests manuels + edge cases
10. `README.md` complet
