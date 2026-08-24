# Internal Note: Enterprise Buyer Requirements (illustrative)

> Sample/synthetic data for the course.

Enterprise adoption of AI coding agents is gated less by raw capability and more
by trust and control. Recurring requirements from buyer conversations:

## Security & compliance
- SOC 2 Type II, and often ISO 27001.
- No training on customer code by default; clear data-retention controls.
- VPC / on-prem / self-hosted deployment options for regulated industries.

## Legal
- IP indemnification for generated code.
- Clear licensing provenance to avoid copyleft contamination.

## Operational
- SSO/SCIM, role-based access control, per-repo permissions.
- Audit logs: who ran what agent, against which repo, with what result.
- Cost controls and usage caps per team.

## Reliability & review
- Every agent change flows through normal code review and CI.
- Deterministic, reproducible behaviour is preferred over "magic".
- Human-in-the-loop approval for anything touching production or secrets.

## Implication for a new entrant
A startup entering this market must treat **governance and reliability as
first-class product**, not as an afterthought bolted onto a model wrapper.
