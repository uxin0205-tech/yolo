# Domain Docs

This repository uses a single-context domain documentation layout.

## Before exploring

Read these files when they exist:

- `CONTEXT.md` at the repository root.
- Relevant architectural decisions under `docs/adr/`.

If these files do not exist, proceed silently. Domain-modeling workflows create them when terminology or architectural decisions need to be recorded.

## Layout

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── AGENTS.md
```

## Vocabulary

Use terminology defined in `CONTEXT.md`. Avoid introducing synonyms that conflict with its glossary.

## Architectural decisions

If proposed work conflicts with an existing ADR, identify the conflict explicitly instead of silently overriding the decision.
