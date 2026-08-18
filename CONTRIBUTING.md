# Contributing

Thanks for helping make LLM lifecycle orchestration simpler without hiding backend constraints.

## Development setup

```bash
git clone https://github.com/aalvsz/llm-lifecycle.git
cd llm-lifecycle
uv sync --extra dev --extra reference
```

Run the local gates before opening a pull request:

```bash
uv run ruff check src tests examples
uv run mypy src
uv run pytest -q
uv build
```

## Compatibility changes

Backend changes should include a focused contract test that pins the native configuration key, command-line
argument, checkpoint boundary, or failure mode being changed. Importing NeMo AutoModel, Megatron Bridge, or
NeMo RL must remain optional for plan compilation.

GPU results must identify the exact backend revision, image, hardware, topology, command, checkpoint, and
terminal outcome. A dry run, CPU test, parser check, or job-started signal is not GPU execution evidence.

## Pull requests

Keep changes narrow and explain:

- the user-visible behavior;
- which backend contract is affected;
- how the change was tested;
- what remains unverified.

Do not commit model weights, datasets, credentials, generated run directories, or backend authentication state.
