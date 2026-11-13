"""Adaptación Python de cafs_matlab_classification_v6.m.

Conserva construcción CA binaria de fuerza 2, búsqueda iterativa,
penalización de compacidad, CV externa/interna, HPO y tablas de estabilidad.
Las implementaciones sklearn no son numéricamente idénticas a MATLAB.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
from math import comb, ceil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class Options:
    Patience: int = 3
    MinImprovement: float = 1e-4
    MinFeatures: int = 5
    MaxIterations: int = 50
    UseHPO: bool = True
    RandomSeed: int = 42
    Preprocessing: str = 'none'
    RegionWidth: float = 25
    TargetFeatures: int = 10
    ExactFeatures: int = 0  # 0: CAFS original; >0: cardinalidad exacta
    CompactnessPenalty: float = 0.10
    ExcessPenaltyPower: float = 2.0
    AdaptivePenaltyAlpha: float = 2.0

def CAN_2_k_2(k):
    if k < 1: raise ValueError('k debe ser positivo')
    n = 1
    while True:
        n += 1
        if comb(n - 1, ceil(n / 2)) >= k:
            return n

def generate_CA_2_k_2(k):
    n = CAN_2_k_2(k)
    ca = np.zeros((n, k), dtype=np.uint8)
    for c, positions in enumerate(combinations(range(1, n), ceil(n / 2))):
        if c == k: break
        ca[list(positions), c] = 1
    return ca

def verify_CA_2(ca):
    missing = []
    for i, j in combinations(range(ca.shape[1]), 2):
        pairs = {tuple(row) for row in ca[:, [i, j]]}
        if len(pairs) != 4: missing.append((i, j, pairs))
    return missing

def apply_spectral_preprocessing(X, preprocessing='none'):
    x = np.asarray(X, dtype=float).copy()
    p = preprocessing.lower()
    if p.startswith('snv'):
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, ddof=1, keepdims=True)
        sd[~np.isfinite(sd) | (sd == 0)] = 1
        x = (x - mu) / sd
    if 'sg' in p:
        deriv = 1 if p.endswith('sg1') else 2
        if x.shape[1] < 11: raise ValueError('SG requiere al menos 11 bandas')
        x = savgol_filter(x, window_length=11, polyorder=2, deriv=deriv, axis=1, mode='interp')
    elif p not in ('none', 'snv'):
        raise ValueError(f'Preprocesamiento no soportado: {p}')
    return x

def _classifier(name, seed=42):
    name = name.lower()
    if name == 'nb': return GaussianNB()
    if name == 'knn': return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    if name == 'svm': return make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    if name == 'tree': return DecisionTreeClassifier(random_state=seed)
    if name == 'rf': return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    raise ValueError('Modelo no soportado: nb, knn, svm, tree o rf')

def _metrics(y, yp, labels):
    return dict(accuracy=accuracy_score(y, yp), macroF1=f1_score(y, yp, labels=labels, average='macro', zero_division=0),
                macroPrecision=precision_score(y, yp, labels=labels, average='macro', zero_division=0),
                macroRecall=recall_score(y, yp, labels=labels, average='macro', zero_division=0))

def compute_selection_score(mean_f1, p, k, iteration, opts):
    lam = opts.CompactnessPenalty * (1 + opts.AdaptivePenaltyAlpha * (iteration - 1) / max(1, opts.MaxIterations))
    penalty = lam * max(0, (p - opts.TargetFeatures) / max(1, k)) ** opts.ExcessPenaltyPower
    return mean_f1 - penalty, penalty, lam

def _cv_metrics(x, y, name, folds, labels, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = []
    for tr, te in cv.split(x, y):
        mdl = _classifier(name, seed)
        mdl.fit(x[tr], y[tr])
        out.append(_metrics(y[te], mdl.predict(x[te]), labels))
    return {k: np.array([row[k] for row in out]) for k in out[0]}

def _cafs_exact_selection(x, y, wavelengths, model_name, inner_folds, opts, labels):
    """CAFS guiado por covering arrays con cardinalidad final exacta.

    Reduce por subconjuntos CA mientras sean viables; cuando los candidatos
    CA no permiten avanzar, elimina una banda por paso mediante CV interna.
    No realiza truncamiento arbitrario de un conjunto de mayor tamaño.
    """
    target = int(opts.ExactFeatures)
    n_features = x.shape[1]
    if not 1 <= target <= n_features:
        raise ValueError(f'ExactFeatures debe estar entre 1 y {n_features}')

    current = np.arange(n_features, dtype=int)
    history = []
    last_ca = None
    cache = {}
    iteration = 0

    def evaluate(idx, iteration_number, total):
        key = tuple(idx.tolist())
        if key not in cache:
            cache[key] = _cv_metrics(x[:, idx], y, model_name,
                                     inner_folds, labels, opts.RandomSeed)
        m = cache[key]
        f1 = float(np.mean(m['macroF1']))
        acc = float(np.mean(m['accuracy']))
        score, penalty, lam = compute_selection_score(
            f1, len(idx), total, iteration_number, opts)
        return dict(idx=idx.copy(), rank=(score, f1, acc), score=score,
                    f1=f1, acc=acc, penalty=penalty, lam=lam, metrics=m)

    # MaxIterations controla la búsqueda CA; la eliminación final no se
    # interrumpe por paciencia, para garantizar exactamente target bandas.
    while len(current) > target:
        iteration += 1
        before = len(current)
        candidate_sets = []
        method = 'greedy-backward'

        if iteration <= opts.MaxIterations and before > target + 1:
            ca = generate_CA_2_k_2(before)
            if verify_CA_2(ca):
                raise RuntimeError('Covering array sin cobertura completa')
            last_ca = ca
            for mask in ca.astype(bool):
                idx = current[mask]
                if target <= len(idx) < before:
                    candidate_sets.append(idx)
            if candidate_sets:
                method = 'covering-array'

        if not candidate_sets:
            candidate_sets = [np.delete(current, j) for j in range(before)]

        best = None
        for idx in candidate_sets:
            item = evaluate(idx, iteration, before)
            if best is None or item['rank'] > best['rank']:
                best = item

        current = best['idx']
        history.append(dict(
            iteration=iteration, method=method,
            numFeaturesBefore=before, numFeaturesAfter=len(current),
            selectedFeatureIndices=current.tolist(),
            selectedWavelengths=np.asarray(wavelengths)[current].tolist(),
            meanAccuracy=best['acc'], meanMacroF1=best['f1'],
            stdMacroF1=float(np.std(best['metrics']['macroF1'], ddof=1)),
            meanMacroPrecision=float(np.mean(best['metrics']['macroPrecision'])),
            meanMacroRecall=float(np.mean(best['metrics']['macroRecall'])),
            selectionScore=best['score'], featurePenalty=best['penalty'],
            adaptiveLambda=best['lam']))
        print(f"CAFS cardinalidad fija iter {iteration}: {before} -> {len(current)} "
              f"bandas ({method}); F1={best['f1']:.3f}")

    final = evaluate(current, max(iteration, 1), len(current))
    return dict(indices=current, wavelengths=np.asarray(wavelengths)[current],
                history=history, bestInternalMacroF1=final['f1'],
                lastGeneratedCA=last_ca)


def cafs_inner_selection_v2(x, y, wavelengths, model_name, inner_folds, opts, labels):
    if getattr(opts, "ExactFeatures", 0):
        return _cafs_exact_selection(x, y, wavelengths, model_name, inner_folds, opts, labels)
    current = np.arange(x.shape[1]); best_idx = current.copy()
    best_score = -np.inf; best_f1 = -np.inf; previous = -np.inf
    no_improve = 0; history = []; last_ca = None
    for iteration in range(1, opts.MaxIterations + 1):
        k = len(current)
        last_ca = generate_CA_2_k_2(k)
        if verify_CA_2(last_ca): raise RuntimeError('Covering array sin cobertura completa')
        candidate = None
        for mask in last_ca.astype(bool):
            idx = current[mask]
            if not len(idx): continue
            m = _cv_metrics(x[:, idx], y, model_name, inner_folds, labels, opts.RandomSeed)
            f1 = float(np.mean(m['macroF1'])); acc = float(np.mean(m['accuracy']))
            score, penalty, lam = compute_selection_score(f1, len(idx), k, iteration, opts)
            rank = (score, f1, acc)
            if candidate is None or rank > candidate['rank']:
                candidate = dict(rank=rank, idx=idx, score=score, f1=f1, acc=acc,
                                 penalty=penalty, lam=lam, metrics=m)
        if candidate is None: break
        if (candidate['score'], candidate['f1']) > (best_score, best_f1):
            best_score, best_f1 = candidate['score'], candidate['f1']
            best_idx = candidate['idx'].copy()
        improvement = candidate['score'] - previous
        no_improve = 0 if improvement > opts.MinImprovement else no_improve + 1
        history.append(dict(iteration=iteration, numFeaturesBefore=k, numFeaturesAfter=len(candidate['idx']),
                            coveringArrayRows=last_ca.shape[0], coveringArrayColumns=k,
                            selectedFeatureIndices=candidate['idx'].tolist(),
                            selectedWavelengths=np.asarray(wavelengths)[candidate['idx']].tolist(),
                            meanAccuracy=candidate['acc'], meanMacroF1=candidate['f1'],
                            stdMacroF1=float(np.std(candidate['metrics']['macroF1'], ddof=1)),
                            meanMacroPrecision=float(np.mean(candidate['metrics']['macroPrecision'])),
                            meanMacroRecall=float(np.mean(candidate['metrics']['macroRecall'])),
                            selectionScore=candidate['score'], featurePenalty=candidate['penalty'],
                            adaptiveLambda=candidate['lam'], improvement=improvement, noImproveCount=no_improve))
        current = candidate['idx']
        previous = max(previous, candidate['score'])
        print(f"CAFS iter {iteration}: {k} -> {len(current)} bandas; F1={candidate['f1']:.4f}; score={candidate['score']:.4f}")
        if len(current) <= opts.MinFeatures or no_improve >= opts.Patience or len(current) <= 1: break
    return dict(indices=best_idx, wavelengths=np.asarray(wavelengths)[best_idx],
                history=history, bestInternalMacroF1=best_f1, lastGeneratedCA=last_ca)

def _optimized_classifier(x, y, name, folds, seed):
    base = _classifier(name, seed)
    grids = {
        'nb': {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]},
        'knn': {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7], 'kneighborsclassifier__weights': ['uniform', 'distance']},
        'svm': {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.01, 0.1, 1]},
        'tree': {'min_samples_leaf': [1, 2, 5, 10], 'max_leaf_nodes': [None, 10, 30, 100]},
        'rf': {'max_features': ['sqrt', 0.5, 1.0], 'min_samples_leaf': [1, 2, 5]},
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(base, grids[name.lower()], scoring='f1_macro', cv=cv, n_jobs=1, refit=True)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_

def CAFS_V6(X, y, wavelengths=None, modelName='nb', outerFolds=5,
                                   innerFolds=5, hpoFolds=3, options=None, output_dir='.'):
    opts = Options(**(options or {})) if isinstance(options, dict) or options is None else options
    x = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if x.ndim != 2 or x.shape[0] != len(y): raise ValueError('X e y deben tener el mismo número de muestras')
    wavelengths = np.arange(1, x.shape[1]+1) if wavelengths is None else np.asarray(wavelengths).ravel()
    if len(wavelengths) != x.shape[1]: raise ValueError('Número de longitudes de onda incorrecto')
    labels = np.unique(y)
    if opts.ExactFeatures and not 1 <= opts.ExactFeatures <= x.shape[1]:
        raise ValueError('ExactFeatures fuera del número de bandas')
    x = apply_spectral_preprocessing(x, opts.Preprocessing)
    cv = StratifiedKFold(n_splits=outerFolds, shuffle=True, random_state=opts.RandomSeed)
    rows = []; fold_results = []; selected = []; last_ca = None
    for fold, (tr, te) in enumerate(cv.split(x, y), 1):
        print(f'Pliegue externo {fold}/{outerFolds}')
        selection = cafs_inner_selection_v2(x[tr], y[tr], wavelengths, modelName, innerFolds, opts, labels)
        idx = selection['indices']; selected.append(idx)
        last_ca = selection['lastGeneratedCA']
        if opts.UseHPO:
            model, params = _optimized_classifier(x[tr][:, idx], y[tr], modelName, hpoFolds, opts.RandomSeed)
        else:
            model = _classifier(modelName, opts.RandomSeed).fit(x[tr][:, idx], y[tr]); params = {}
        pred = model.predict(x[te][:, idx]); m = _metrics(y[te], pred, labels)
        rows.append(dict(OuterFold=fold, Accuracy=m['accuracy'], MacroF1=m['macroF1'],
                         MacroPrecision=m['macroPrecision'], MacroRecall=m['macroRecall'], NumSelectedFeatures=len(idx)))
        fold_results.append(dict(outerFold=fold, selectedFeatureIndices=idx.tolist(),
                                 selectedWavelengths=wavelengths[idx].tolist(), innerHistory=selection['history'],
                                 bestHyperparameters=params, yTrue=y[te], yPred=pred,
                                 confusionMatrix=confusion_matrix(y[te], pred, labels=labels), **m))
    summary = pd.DataFrame(rows)
    stability = pd.DataFrame(dict(FeatureIndex=np.arange(1, x.shape[1]+1), Wavelength=wavelengths,
                                  SelectionCount=[sum(j in s for s in selected) for j in range(x.shape[1])]))
    stability['SelectionFrequency'] = stability['SelectionCount'] / outerFolds
    stability = stability.sort_values(['SelectionCount', 'FeatureIndex'], ascending=[False, True])
    region_width = opts.RegionWidth
    region = np.floor((wavelengths - np.min(wavelengths)) / region_width).astype(int)
    region_rows = []
    for r in np.unique(region):
        positions = np.flatnonzero(region == r)
        count = sum(bool(np.intersect1d(s, positions).size) for s in selected)
        region_rows.append(dict(RegionStart=float(np.min(wavelengths[positions])),
                                RegionEnd=float(np.max(wavelengths[positions])),
                                SelectionCount=count, SelectionFrequency=count/outerFolds))
    regions = pd.DataFrame(region_rows).sort_values('SelectionCount', ascending=False)
    selected_table = pd.DataFrame([dict(OuterFold=i+1, SelectedWavelengths=';'.join(map(str, wavelengths[s]))) for i, s in enumerate(selected)])
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'cafs_v6_classification_summary.csv', index=False)
    stability.to_csv(out/'cafs_v6_feature_wavelength_stability.csv', index=False)
    regions.to_csv(out/'cafs_v6_region_stability.csv', index=False)
    selected_table.to_csv(out/'cafs_v6_selected_wavelengths_by_fold.csv', index=False)
    if last_ca is not None: np.savetxt(out/'coveringArray_last_generated.csv', last_ca, delimiter=',', fmt='%d')
    result = dict(modelName=modelName, options=asdict(opts), foldResults=fold_results,
                  summaryTable=summary, featureStabilityTable=stability, regionStabilityTable=regions,
                  selectedWavelengthTable=selected_table, lastGeneratedCoveringArray=last_ca)
    for col in ['Accuracy', 'MacroF1', 'MacroPrecision', 'MacroRecall']:
        result['mean'+col] = float(summary[col].mean())
        result['std'+col] = float(summary[col].std(ddof=1))
    return result

