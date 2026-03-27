# Bayesian Inference and Bayesian Statistics

This repository contains a Quarto course website on Bayesian inference, Bayesian statistics, computation, modeling, diagnostics, and workflow.

## Build locally

1. Install [Quarto](https://quarto.org/).
2. Render the site:

```bash
quarto render
```

The site output is written to `_site/`.

## Optional runtime setup

- R examples: `Rscript install.R`
- Python/JAX examples: `python -m pip install -r requirements.txt`

Stan examples are designed for `cmdstanr`; set `INSTALL_CMDSTAN=true` before running `install.R` if you also want to install CmdStan.

## License

- Course content: `LICENSE-CONTENT.md` (`CC BY-NC-SA 4.0`)
- Code examples and software artifacts: `LICENSE-CODE` (`MIT`)

The site structure and course-front framing were inspired in part by Claire David's course site:
<https://clairedavid.github.io/intro_to_ml/intro.html>

## Repository structure

```text
.
├── _quarto.yml
├── index.qmd
├── course-blueprint.qmd
├── syllabus.qmd
├── chapter-outline.qmd
├── parts/
├── chapters/
├── appendix/
├── bibliography/
├── labs/
├── styles/
└── .github/workflows/
```
