# NourishBot deployment contract

NourishBot's public Next.js application targets Vercel. Configure the Vercel project
Root Directory as `frontend`; its version-controlled configuration is
[`frontend/vercel.json`](../frontend/vercel.json).

The present FastAPI process also starts an outbox loop and Telegram polling. Scheduled-order
activation and retention use separate Celery worker and beat processes. These long-running
services remain on an always-on container runtime with managed PostgreSQL and Redis until the
full-Vercel migration work in [`docs/guides/VERCEL_DEPLOYMENT.md`](../docs/guides/VERCEL_DEPLOYMENT.md)
is complete.

Azure deployment configuration is intentionally not part of this repository.
