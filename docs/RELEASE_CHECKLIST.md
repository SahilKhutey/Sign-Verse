# Release Checklist

## Pre‑Release

- [ ] Run unit tests: `pytest`
- [ ] Run lint: `ruff check .`
- [ ] Train or verify NLP models are available in `models/`
- [ ] Verify `/health` and `/health/ready` for backend + api_server
- [ ] Confirm `.env` values for production (no `*` in `ALLOWED_ORIGINS`)
- [ ] Confirm models and datasets are stored outside git

## Staging

- [ ] Deploy to staging environment
- [ ] Run smoke tests for all inference endpoints
- [ ] Validate auth + translation history flows
- [ ] Validate Android + Web clients against staging

## Production

- [ ] Tag release
- [ ] Deploy with versioned model registry artifacts
- [ ] Enable monitoring + alerting
- [ ] Verify post‑deploy health and latency
