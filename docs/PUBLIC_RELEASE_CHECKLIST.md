# Public-Release Checklist

Use this checklist before making the repository public, submitting it for assessment, or
sharing it outside the people already authorized to access it.

## Contract and ownership

- [ ] Review the NDA's definition of confidential information.
- [ ] Review employment, founder, contractor, and invention-assignment clauses.
- [ ] Confirm which code and assets you personally have permission to publish.
- [ ] Obtain qualified legal advice where ownership or permitted use is unclear.

## Repository content

- [ ] Remove or rewrite legacy strategy and commercial roadmap documents.
- [ ] Remove real company, customer, vendor, and employee information.
- [ ] Remove workbooks, exports, screenshots, QR codes, and PDFs not created for this demo.
- [ ] Replace all menus and operational records with synthetic fixtures.
- [ ] Remove live third-party marketplace and payment configuration.
- [ ] Verify that Git history does not expose removed confidential content.

## Secrets and access

- [ ] Run a history-aware secret scan.
- [ ] Revoke and rotate any credential that was ever committed or shared.
- [ ] Confirm `.env`, OAuth caches, cloud credentials, and private keys are ignored.
- [ ] Test with fresh demo-only accounts.

## Product claims

- [ ] Describe NourishBot as a canteen pre-order and pickup portfolio project.
- [ ] Do not present planned features as implemented.
- [ ] Do not claim medical, dietary, allergy, security, or production guarantees.
- [ ] Clearly label synthetic data and mocked integrations.

## Final technical validation

- [ ] Frontend production build passes.
- [ ] Backend unit and API tests pass.
- [ ] Database migrations run from an empty database.
- [ ] Demo seed data contains no real personal or business information.
- [ ] Public documentation matches the actual application behavior.
