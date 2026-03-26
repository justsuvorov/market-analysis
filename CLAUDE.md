# CLAUDE.md — Market Analysis Project

## Methodology: Spec-Driven Development (SDD)

**All code changes begin with a specification update. Never modify code without an approved spec first.**

### SDD Workflow

1. Identify the change needed
2. Update or create the relevant spec in `/specs` (Markdown, OpenAPI YAML, or JSON Schema)
3. **Present the spec to the user and wait for approval**
4. Only after approval: generate/update code from the spec
5. Commit spec and code together

If a user asks for a code change without a spec, respond by drafting the spec first.

---

## Project Structure

```
market-analysis/
├── specs/                  # Specifications (source of truth)
│   ├── marketing_research_v1.md   # Main research spec
│   ├── schemas/            # JSON Schema / OpenAPI schemas
│   └── agents/             # Per-agent specs
├── src/
│   ├── agents/             # Pydantic-AI agent implementations
│   ├── models/             # Pydantic data models (generated + manual)
│   └── ml/                 # CatBoost / GLM training & inference
├── data/                   # Raw and processed datasets (gitignored)
├── notebooks/              # Exploratory analysis
├── tests/
└── CLAUDE.md
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Agent framework | [Pydantic-AI](https://ai.pydantic.dev/) |
| Data models | Pydantic v2 |
| ML (classification/regression) | CatBoost |
| ML (interpretable linear) | GLM (statsmodels / sklearn) |
| Code generation from schemas | `datamodel-codegen` |
| Python version | 3.11+ |

---

## Code Generation from Schemas

To regenerate Pydantic models from a JSON Schema:

```bash
datamodel-codegen \
  --input specs/schemas/<schema_file>.json \
  --input-file-type jsonschema \
  --output src/models/<output_file>.py \
  --use-annotated \
  --field-constraints
```

To regenerate from OpenAPI:

```bash
datamodel-codegen \
  --input specs/schemas/openapi.yaml \
  --input-file-type openapi \
  --output src/models/
```

**Always run this command after approving a schema change. Do not hand-edit generated files.**

---

## Agent Rules (Pydantic-AI)

- Every agent lives in `src/agents/<agent_name>.py`
- Every agent has a corresponding spec in `specs/agents/<agent_name>.md`
- All inter-agent data exchange must use typed Pydantic models from `src/models/`
- Agent result types must be explicitly annotated: `Agent[InputModel, OutputModel]`
- No raw `dict` passing between agents

---

## ML Rules

- CatBoost is the default for classification and regression tasks
- GLM (statsmodels) is used when business interpretability of coefficients is required
- Every model must include a feature importance or coefficient table in its output
- Model configs (hyperparameters) live in `specs/` or `src/ml/configs/`
- Explain every feature in business terms in the spec before training

---

## Specs Format

Each spec in `/specs` must include:

```markdown
## Status
Draft | Review | Approved | Deprecated

## Purpose
What business problem this solves.

## Inputs
Pydantic model name + field descriptions.

## Outputs
Pydantic model name + field descriptions.

## Agents involved
List of agents and their roles.

## ML approach
Algorithm, target variable, key features, interpretability notes.

## Schema reference
Path to JSON Schema / OpenAPI file (if applicable).

## Open questions
Items pending user confirmation.
```

---

## What Requires User Approval Before Proceeding

- Any new or changed spec
- Any new agent design
- Changes to Pydantic model schemas that affect inter-agent contracts
- Choice of ML algorithm or feature set
- Changes to this CLAUDE.md
