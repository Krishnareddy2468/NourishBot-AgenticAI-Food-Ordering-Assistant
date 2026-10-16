.PHONY: install dev build lint test phase8-up phase8-down phase8-migrate phase8-migration-rehearsal phase8-integration phase8-backup-restore phase8-unit

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

phase8-up:
	docker compose -f infra/compose/phase8-validation.yml up -d --wait

phase8-down:
	docker compose -f infra/compose/phase8-validation.yml down

phase8-migrate:
	cd backend && DATABASE_URL=postgresql+asyncpg://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 DATABASE_URL_SYNC=postgresql+psycopg2://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 .venv/bin/alembic upgrade head

phase8-migration-rehearsal:
	cd backend && DATABASE_URL=postgresql+asyncpg://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 DATABASE_URL_SYNC=postgresql+psycopg2://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 .venv/bin/alembic downgrade base
	cd backend && DATABASE_URL=postgresql+asyncpg://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 DATABASE_URL_SYNC=postgresql+psycopg2://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 .venv/bin/alembic upgrade head

phase8-integration:
	cd backend && PHASE8_INTEGRATION=1 PHASE8_POSTGRES_DSN=postgresql://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 PHASE8_REDIS_URL=redis://localhost:56379/0 DATABASE_URL=postgresql+asyncpg://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 DATABASE_URL_SYNC=postgresql+psycopg2://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 REDIS_URL=redis://localhost:56379/0 TELEGRAM_BOT_ENABLED=false .venv/bin/pytest -q tests/integration

phase8-backup-restore:
	backend/.venv/bin/python scripts/postgres_backup_restore.py backup --source postgresql://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 --output /tmp/nourishbot-phase8.backup
	backend/.venv/bin/python scripts/postgres_backup_restore.py drill --source postgresql://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8 --target postgresql://nourishbot:phase8-local-only@localhost:55432/nourishbot_phase8_restore --backup /tmp/nourishbot-phase8.backup --confirm-target-db nourishbot_phase8_restore --report /tmp/nourishbot-phase8-restore-report.json

phase8-unit:
	cd backend && PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q tests/unit
