# Contributing to SignVerse-AI

Thanks for helping improve SignVerse-AI. This repo contains research code, training pipelines, and production services, so we keep changes tightly scoped and well documented.

## Quick Start

1. Create a feature branch from `main`.
2. Install dependencies:
```
python -m pip install -r requirements.txt
```
3. Run tests:
```
pytest
```

## Branching

- Use a short, descriptive branch name such as `feature/text-gloss-training`.
- Keep PRs focused to a single theme.

## Commit Guidelines

- Use present tense: "Add BLEU scoring".
- Keep commits small and explain "why" in the message.

## Code Style

- Python: follow PEP8 where possible.
- Prefer clear, explicit names over abbreviations.
- Avoid adding large binaries (models, datasets, logs).

## Tests

- Add or update tests when behavior changes.
- Run `pytest` before submitting a PR.

## Documentation

- Update `README.md` for user-facing changes.
- Add new training steps to `docs/training_pipeline.md` when relevant.

## Security

- Do not commit secrets or API keys.
- Use environment variables for configuration.

## Reporting Issues

When reporting issues, include:
- Expected behavior
- Actual behavior
- Steps to reproduce
- Logs or stack traces
