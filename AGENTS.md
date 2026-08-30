# Agents

## Markdown

### Tables

Always keep plain text Markdown tables aligned. Use consistent spacing around each cell so that the column separators line up vertically in the source, and update the alignment whenever a cell is added or changed.

### Linting

Run `markdownlint` on every Markdown file you touch and resolve any reported issues before finishing.

checks:

```bash
markdownlint content/**/*.md
markdownlint content/**/*.qmd
```

fixing:

```bash
markdownlint content/**/*.md --fix
markdownlint content/**/*.qmd --fix
```
