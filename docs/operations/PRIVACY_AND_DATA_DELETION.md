# Privacy, Export, Retention, and Deletion Workflow

The public notice is available at `/privacy`. Before public release, the deployment operator must add its legal identity, jurisdiction-specific disclosures, and contact channel after legal review.

## Identity verification

Privacy self-service operates only through an authenticated customer account or an opaque web identity securely linked from an already verified Telegram or WhatsApp channel. A phone number, email assertion, order reference, or support message alone is not sufficient to merge identities or disclose data.

## Purpose-specific consent

Consent decisions are append-only and independently recorded for personalization, location, reminders, and analytics. Withdrawal takes effect for future processing. Location withdrawal also deletes retained saved locations, routes, and discovery history. Transaction and security records remain only where required for accounting, fraud prevention, dispute resolution, or legal duties.

## Export

The customer sends `export my memory` or uses the authenticated privacy API. The export must contain only the resolved customer’s profile, preferences, consent status, and masked linked channels. Support verifies the request ID and delivery channel but never asks for a password, OTP, full payment credential, or MCP token.

## Permanent deletion

1. Customer initiates deletion through the verified channel.
2. System requires the exact confirmation phrase `delete my memory permanently`.
3. The privacy service deletes learned preferences and retained personal memory in a transaction and appends a privacy-request audit record.
4. Location/route records are physically removed when location consent is withdrawn or deletion requires it.
5. Support confirms completion without reproducing deleted content.
6. Any legally retained transaction record is access-restricted and excluded from personalization.

Do not perform ad-hoc SQL deletion. If the automated workflow fails, open a restricted manual-review case, preserve the error and request identifiers, and have the privacy owner approve a tested remediation.

## Retention operations

Celery Beat runs the retention worker. Operators monitor its heartbeat and failures. Current deploy-time limits are defined by `MESSAGE_RETENTION_DAYS`, `LOCATION_RETENTION_DAYS`, and `PROVIDER_RESPONSE_RETENTION_DAYS`. Reducing a limit should trigger prompt pruning; increasing one requires privacy review and a policy-version update.

Monthly evidence should include retention-job success, sampled cutoff verification without customer content, deletion completion counts, unresolved requests, and consent-withdrawal cleanup results.
