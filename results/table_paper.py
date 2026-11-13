"""Overall accuracy and macro-F1 (mean, SD, min, max over 30 hold-outs) from the
_Global sheets of metrics.xlsx. SVM and ANN from the wc_* runs; L-SVM from lsc_m*
(converged, dual=False); LDA from lda/. Writes table_paper.csv (Table 2 of the paper)."""
import pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parent
def hoja(c, m):
    if c == "SVM": d = A/("wc_full_m0_c1" if m == 0 else f"wc_sel_m{m}_c1")
    elif c == "ANN": d = A/("wc_full_m0_c3" if m == 0 else f"wc_sel_m{m}_c3")
    elif c == "L-SVM": d = A/f"lsc_m{m}"
    else: d = A/"lda"
    return pd.read_excel(d/"metrics.xlsx", f"{c}_{m+1}_Global").set_index("Metrica")
filas = []
for m, n in enumerate(["Full spectrum", "FSCNCA", "SFS", "CAFS"]):
    for c in ["SVM", "L-SVM", "LDA", "ANN"]:
        g = hoja(c, m)
        filas.append({"bands": n, "clf": c,
                      "acc": 100*g.loc["Accuracy global", "Media"], "acc_sd": 100*g.loc["Accuracy global", "Desviacion_estandar"],
                      "acc_min": 100*g.loc["Accuracy global", "Minimo"], "acc_max": 100*g.loc["Accuracy global", "Maximo"],
                      "f1": 100*g.loc["F1 macro", "Media"], "f1_sd": 100*g.loc["F1 macro", "Desviacion_estandar"]})
d = pd.DataFrame(filas); d.to_csv(A/"table_paper.csv", index=False)
print(d.round(1).to_string(index=False))
