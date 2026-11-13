# Rice variety discrimination by NIR hyperspectral imaging — analysis code

Code, data and results behind the paper *Rice variety discrimination based on NIR
hyperspectral imaging: impact of variable selection methods*.

The analysis code was written by Wilson Castro Silupu (Universidad Nacional de Frontera,
Peru). `original/` holds it exactly as received; `code/` is the version run for the paper.

## Layout

| Path | Contents |
|---|---|
| `original/` | `ANALISIS_ARROZ.py`, `FUNCIONES.py`, `CAFS.py` as received |
| `code/` | the same three files as run for the paper (changes below) |
| `code/data/profile_totales.mat` | pre-treated mean profiles, 6000 grains × 336 bands, with class labels |
| `code/data/nBandas.mat` | wavelength of each band; bands 47–309 (999–1649 nm) are analyzed |
| `results/*/metrics.xlsx` | per-hold-out and summary metrics of each run |
| `results/table_paper.py`, `table_paper.csv` | Table 2 of the paper, built from the `_Global` sheets |

## Changes from `original/` to `code/`

1. `ANALISIS_ARROZ.py`: `N = 3` → `N = 30` hold-out repetitions (lines 236 and 266);
   classifier list `[1, 2, 3]` → `[1, 2, 3, 4]` (lines 232 and 265).
2. `FUNCIONES.py`, classifier 4 added: `LinearDiscriminantAnalysis()`. Option 2 is a
   linear SVM (`LinearSVC`) and is labelled `L-SVM`; option 4 is labelled `LDA`.
3. `FUNCIONES.py`, option 2: `LinearSVC(dual=True, …)` → `LinearSVC(dual=False, …)`, the
   same model solved in the primal so that it converges (no convergence warnings).

Nothing else was changed. The validation is the code's own: N random stratified 70/30
hold-outs (`train_test_split`, unseeded), so a rerun reproduces the means within their SD,
not to the last digit.

## Running

```bash
pip install numpy scipy pandas matplotlib scikit-learn openpyxl
cd code && python ANALISIS_ARROZ.py      # writes metrics.xlsx, Figuras/, Resultados/
```

The full run (4 band sets × 4 classifiers × 30 hold-outs) is long; for the paper it was split
into one job per band set and classifier (`results/` subfolder names: `wc_full_m0_c<k>`,
`wc_sel_m<m>_c<k>`, `lsc_m<m>`, `lda`; `m` = band set 0 full, 1 FSCNCA, 2 SFS, 3 CAFS;
`k` = classifier), each changing only the classifier and method lists in `ANALISIS_ARROZ.py`.

Python 3.14, scikit-learn 1.9.
