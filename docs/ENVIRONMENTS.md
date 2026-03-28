# Environments

This repo uses simple `.env` profiles to separate dev, staging, and production.

## Files

- `.env.example` (local development template)
- `.env.staging.example`
- `.env.production.example`

## Core Variables

- `ENVIRONMENT`: `development`, `staging`, or `production`
- `SECRET_KEY`: required in prod
- `ALLOWED_ORIGINS`: comma-separated origins (no `*` in prod)
- `INFERENCE_API_URL`: base URL of inference server
- `DATABASE_URL`: database connection string

## Frontend Variables

- `VITE_BACKEND_URL`
- `VITE_INFERENCE_URL`

## Usage

Copy one of the examples to `.env` and adjust values for your environment.
