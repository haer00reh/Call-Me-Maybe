## The very first line must be italicized and read: This project has been created as part of the 42 curriculum by haer-reh!

# Description
this project is about implementing a function calling system using a small llm 

# Instructions
run – Runs the application using uv and executes the src module with the configured Python interpreter.
install – Installs Poetry (if needed) and installs all project dependencies defined in pyproject.toml.
debug – Runs the application in the same way as run, typically intended for debugging or future debugging-specific options.
clean – Removes generated cache directories (__pycache__ and .mypy_cache) to keep the project directory clean.
lint – Checks the code for style issues with flake8 and performs static type checking with mypy using strict type-checking options.
---
**run the project with specific output/input/definition json files** - uv run python -m src --functions_definition <path> --input <path> --output <path>

# Resources
very useful article (really helped)
https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output

short but useful
https://zeroentropy.dev/concepts/constrained-decoding/

useful resource from invidia itself
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Constrained_Decoding/README.html

youtube videos
https://youtu.be/xpvFinvqRCA?si=HUE20qTo-SyDFFsp
https://youtu.be/S-8yr_RibJ4?si=8w3lUKjrwjdbWCsn

## AI usage
ai was barely used in this project, only for flake8 and for generating **test** input and definition json files the resources above were very helpful
---
# Algorithm Explanation

Build a prompt listing the available functions and the user query, then encode it so the model can generate the next token.
Hardcode the fixed JSON skeleton (`{"function":"`, `","parameter":`, etc.) directly — never let the model generate punctuation it might get wrong.
**Function name generation:** loop token-by-token, only allowing identifier-shaped tokens (letters/digits/underscore) or a closing quote, always picking the model's highest-scoring allowed token (greedy decoding), until a `"` closes the string.
Validate the generated name against the known function list — reject if it doesn't match.
**Parameter generation:** loop token-by-token again, tracking `{`/`}` depth, allowing a broader token set, until the braces balance out — then parse the result as JSON.
Normalize numeric types (int vs. float) based on the function's parameter schema.
Return `{function, parameter}`. Repeat for every prompt and save all results.

# Design Decisions

In an early stage of the implementation, the model was constrained to generate one character at a time. This was easy to constrain but slow and unreliable. The current approach lets the model generate multi-character tokens directly, so function names and parameter values are produced in far fewer steps, only constrained on structure (identifier shape, brace depth) rather than individual characters.

# Performance Analysis

The fixed JSON skeleton (`{"function":"`, `","parameter":`, etc.) is hardcoded rather than generated, since these tokens are always the same and don't need a model call to produce. This cuts down the number of inference steps significantly, as only the function name and parameter values actually require the model's judgment.

# Challenges Faced

Debugging required running the model on each test case, which was slow given CPU-only inference — this made iteration on the constraint logic time-consuming.

# Testing Strategy
Test prompts were generated using an AI model, covering a variety of phrasings for each available function. Each prompt was run through the agent, and the output was checked for two things:
Whether the correct function name was selected
Whether the generated parameters (keys and values) matched the expected arguments exactly

Mismatches — wrong function selected, wrong parameter keys, or incorrect value types (e.g. float instead of integer) — were treated as failures and used to guide fixes (e.g. hardcoding parameter keys, adding type-aware numeric normalization).


# Example Usage

See the **Instructions** section above.