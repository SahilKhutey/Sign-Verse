# MVP Scope

This document defines what "MVP‑ready" means for SignVerse‑AI.

## Core MVP Capabilities

1. **Sign → Text → Speech**
   - `api_server` supports sign‑to‑text and sign‑to‑speech.
2. **Speech → Text → Sign**
   - `api_server` supports speech‑to‑sign and text‑to‑sign.
3. **Minimal UI**
   - Web UI and Android client can call live endpoints and display responses.
4. **Basic Auth**
   - Backend supports user register/login and translation history.

## Required MVP Quality Gates

- Inference endpoints return valid outputs for sample inputs.
- Text↔gloss translation works with trained or rule‑based fallback.
- All services have `/health` and `/health/ready`.
- README documents how to run locally.

## Explicit Non‑Goals for MVP

- Full avatar retargeting
- On‑device inference optimization
- Multi‑tenant production scaling
