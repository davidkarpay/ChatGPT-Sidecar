.PHONY: run dev test clean install

# Production run with Gunicorn (Railway deployment)
run:
	gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:${PORT:-8080} --timeout 120

# Development server
dev:
	uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload

# Run tests
test:
	pytest tests/ -v

# Clean Python cache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install dependencies
install:
	pip install -r requirements.txt

# Setup development environment
setup-dev:
	python3 -m venv .venv
	source .venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env