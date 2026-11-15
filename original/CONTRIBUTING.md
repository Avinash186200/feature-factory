# Contributing

## Branch workflow

- Don't work directly on `main`. Create a branch named `intern/<your-name>` (e.g. `intern/your-name`)
  and do all your work there.
- Commit early and often — small commits are easier to review than one giant one at the end.
- Open a pull request into `main` whenever you have something worth a look, even if it's not
  finished. A draft PR is fine, and honestly preferred over going quiet for two weeks.
- Don't merge your own PR.

## Getting set up

1. Clone the repo and create your branch.
2. `pip install -r requirements.txt` inside a virtual environment.
3. Read `docs/project_brief.md` for the full project description.
4. Look at `src/features/example_realized_vol.py` before writing your own feature — it's the
   pattern to follow.

## Ground rules

- **No lookahead.** Never let a feature use information that wouldn't actually have been available
  at that point in time. If you're not sure whether something counts as leakage, ask — it's the
  single most common way this kind of research goes wrong.
- **Test before you believe it.** A feature that looks great and hasn't been checked against a
  naive persistence baseline isn't a result yet.
- **Small, honest features beat one giant clever one.** If a feature is genuinely useless, say so —
  a well-documented negative result is worth more than a hidden one.
