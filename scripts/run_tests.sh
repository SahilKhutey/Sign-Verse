#!/bin/bash

# SignVerse Test Runner
set -e

echo "Running SignVerse tests..."

# Run unit tests
echo "=== Unit Tests ==="
python -m pytest tests/unit/ -v --cov=./ --cov-report=term-missing

# Run integration tests
echo "=== Integration Tests ==="
python -m pytest tests/integration/ -v

# Run pipeline tests
echo "=== Pipeline Tests ==="
python -m pytest tests/pipeline/ -v

# Generate coverage report
echo "=== Coverage Report ==="
python -m pytest --cov=./ --cov-report=html:coverage_report

echo "All tests completed successfully!"
