## PR Summary
This PR upgrades the core translation pipeline and training stack.

### What changed
- Added NLP text↔gloss transformer training with BLEU/WER evaluation and curriculum learning.
- Added dataset builder for larger text↔gloss pairs.
- De-mocked core translation pipeline (sign↔text↔speech).
- Added backend hardening, auth refresh tokens, and health checks.
- Wired web UI to inference/backend APIs.
- Added CI + contribution docs and updated README.

### Testing
- [ ] `pytest`
- [ ] `ruff check .`
- [ ] Manual smoke: `api_server` `/health`, `/translate/text-to-sign`, `/translate/sign-to-speech`

### Notes
- Large artifacts should remain out of git (`models/`, `datasets/`, `training-data/`).
