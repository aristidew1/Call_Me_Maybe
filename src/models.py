from typing import Literal
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Definition of a parameter or return type of a function."""

    type: Literal["number", "string", "boolean", "integer"]
    max_tokens: int = 20


class FunctionDef(BaseModel):
    """Complete definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class Prompt(BaseModel):
    """A natural language prompt to be translated into a function call."""

    prompt: str


class FunctionCall(BaseModel):
    """A resolved function call with its name and concrete argument values."""

    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]
