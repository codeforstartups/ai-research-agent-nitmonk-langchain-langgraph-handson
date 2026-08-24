# Internal Note: Competitive Landscape (illustrative)

> Sample/synthetic data for the course. Positioning is illustrative.

## Categories and representative players

- **Inline assistants**: broad, well-funded incumbents bundled with existing
  developer platforms. Strong distribution, thin differentiation.
- **Agentic IDEs**: fast-moving startups competing on the quality of the
  edit-apply-test loop and the editor experience.
- **Autonomous agents**: ambitious entrants marketing "AI software engineer"
  autonomy; demos impressive, production reliability still contested.

## Moats that appear to matter

1. **Distribution** — being where developers already work (editor, code host,
   CI) beats a standalone tool.
2. **Reliability at task-completion** — measured by whether generated changes
   pass tests and review, not by benchmark leaderboards.
3. **Enterprise trust** — SOC2, IP indemnity, on-prem/VPC deployment, audit
   logs, and admin controls.
4. **Context depth** — how well the tool grounds itself in the customer's actual
   codebase, docs, and conventions.

## Weaknesses common across the field

- Overclaiming autonomy relative to real reliability.
- Weak handling of large, messy, legacy repositories.
- Thin observability into *why* an agent made a change.
