# Compiling a LaTeX report to PDF

How to build the technical-report PDFs under [`reports/`](../reports/) (e.g.
`reports/2026-07-06-frozen-vjepa2-physics-probe-9pager.tex`). These are plain
`article`-class LaTeX documents with an **inline** bibliography
(`\thebibliography`, not BibTeX), so the build is just `pdflatex` run a few
times — no `bibtex`/`biber` step.

## Prerequisites

- **A TeX distribution** providing `pdflatex` (TeX Live 2022+ is known to work).
  Check with:
  ```bash
  pdflatex --version
  ```
  On Debian/Ubuntu, install with:
  ```bash
  sudo apt-get install texlive-latex-recommended texlive-latex-extra \
                       texlive-fonts-recommended texlive-science
  ```
  (`texlive-latex-extra` supplies `titlesec`, `enumitem`, `microtype`;
  `texlive-science` supplies `amsmath`/`amssymb`.)
- **The figures must already exist.** The document `\includegraphics` these
  PDFs from `reports/figures/`:
  - `figures/probe_layer_sweep.pdf`
  - `figures/oracle_gate.pdf`
  - `figures/ecc46_readout.pdf`

  They are regenerated from data by
  [`reports/make_probe_figures.py`](../reports/make_probe_figures.py) if
  missing or stale.

## Build (the normal case)

Run from inside `reports/` so the relative `figures/...` paths resolve:

```bash
cd reports
pdflatex -interaction=nonstopmode -halt-on-error 2026-07-06-frozen-vjepa2-physics-probe-9pager.tex
pdflatex -interaction=nonstopmode -halt-on-error 2026-07-06-frozen-vjepa2-physics-probe-9pager.tex
```

The output is `2026-07-06-frozen-vjepa2-physics-probe-9pager.pdf` in the same
directory.

**Why run it twice.** The first pass writes an `.aux` file recording where each
section, table, figure, and `\bibitem` landed; the second pass resolves the
`\ref`, `\cite`, and page numbers against it. A single pass leaves `??` in
place of cross-references. If you have changed section/figure ordering a lot,
run it a **third** time to be safe — the page count and all references must be
stable between the last two passes.

One-liner (three passes, quiet):

```bash
cd reports
for i in 1 2 3; do \
  pdflatex -interaction=nonstopmode -halt-on-error \
    2026-07-06-frozen-vjepa2-physics-probe-9pager.tex >/dev/null; \
done
```

To make it generic for any report, replace the filename with `"$NAME.tex"`.

## Optional: regenerate the figures first

Only needed if the plots are missing or the underlying numbers changed:

```bash
cd reports
python3 make_probe_figures.py          # writes figures/*.pdf and *.png
```

## Optional: visual QA by rendering pages to PNG

Handy for eyeballing whitespace/layout without opening a PDF viewer (needs
`poppler-utils`):

```bash
cd reports
pdftoppm -png -r 120 2026-07-06-frozen-vjepa2-physics-probe-9pager.pdf /tmp/page
# -> /tmp/page-01.png, /tmp/page-02.png, ...
```

## Confirm the result

```bash
grep "Output written" 2026-07-06-frozen-vjepa2-physics-probe-9pager.log
# -> Output written on ...-9pager.pdf (9 pages, ...)
```

A clean build reports no `Overfull \vbox` and no `Overfull \hbox` with a large
overflow (a few points is harmless). To scan the log:

```bash
grep -iE "overfull|undefined|latex error" 2026-07-06-frozen-vjepa2-physics-probe-9pager.log
```

## Cleanup

`pdflatex` leaves `.aux`, `.log`, `.out` scratch files next to the PDF. They
are safe to delete and are regenerated on the next build:

```bash
cd reports
rm -f 2026-07-06-frozen-vjepa2-physics-probe-9pager.{aux,log,out}
```

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `??` shown for a reference or citation | Only one pass ran — run `pdflatex` again. |
| `File 'figures/....pdf' not found` | Figures not built — run `python3 make_probe_figures.py`, or `cd reports` so the relative path resolves. |
| `! LaTeX Error: File 'titlesec.sty' not found` (or `enumitem`, `microtype`) | Missing package — install `texlive-latex-extra`. |
| Build hangs waiting for input on an error | You omitted `-interaction=nonstopmode`; add it (and `-halt-on-error` to stop on the first error). |
| Page count/refs differ between the last two passes | Run one more pass until they stabilize. |
