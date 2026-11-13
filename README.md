# Rice variety discrimination by NIR hyperspectral imaging — analysis code

Code, data and results behind the paper *Rice variety discrimination based on NIR
hyperspectral imaging: impact of variable selection methods*.

The analysis code was written by Wilson Castro Silupu (Universidad Nacional de Frontera, Peru).

## Layout

| Path | Contents |
|---|---|
| `ANALISIS_ARROZ.py`, `FUNCIONES.py`, `CAFS.py` | analysis code (W. Castro, UNF), as run for the paper |
| `data/profile_totales.mat` | pre-treated mean profiles, 6000 grains, with class labels |
| `data/nBandas.mat` | wavelength of each band; bands 47–309 (999–1649 nm) are analyzed |
| `results/*/metrics.xlsx` | metrics of each run, per hold-out and summary |
| `results/*/Figuras/` | figures written by the code |
| `results/table_paper.py`, `table_paper.csv` | Table 2 of the paper |
| `figures/Figuras - Articulo.vsdx` | Visio drawings of the figures (UNF) |
| `figures/fig_workflow_v10.svg`, `fig_models_v10.svg` | Figures 1 and 2 of the paper, editable in Visio |
| `figures/*.png` | the figures of the paper |

## Settings used for the paper

- 30 hold-out repetitions (70 % training, 30 % test, stratified by variety).
- Four classifiers: 1 SVM (RBF), 2 L-SVM (linear SVM, solved in the primal so it converges), 3 ANN, 4 LDA.
  Lines changed for this are marked `Correccion` in `FUNCIONES.py`.

## Running

```bash
pip install numpy scipy pandas matplotlib scikit-learn openpyxl
python ANALISIS_ARROZ.py      # writes metrics.xlsx, Figuras/, Resultados/
```

The full run (4 band sets × 4 classifiers × 30 hold-outs) is long; for the paper it was split
into one job per band set and classifier (`results/` subfolder names: `wc_full_m0_c<k>`,
`wc_sel_m<m>_c<k>`, `lsc_m<m>`, `lda`; `m` = band set 0 full, 1 FSCNCA, 2 SFS, 3 CAFS;
`k` = classifier), each changing only the classifier and method lists in `ANALISIS_ARROZ.py`.

Python 3.14, scikit-learn 1.9.
