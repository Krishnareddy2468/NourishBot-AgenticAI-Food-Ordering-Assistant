.PHONY: install dev build lint test

install:
	pnpm install
	pip install -r backend/requirements.txt

dev:
	pnpm dev

build:
	pnpm build

lint:
	pnpm lint

test:
	pnpm test
	pytest backend/tests --tb=short
