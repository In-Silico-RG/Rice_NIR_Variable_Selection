"""
Funciones implementadas para el uso del script

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from sklearn.metrics import confusion_matrix
#from matplotlib.patches import Rectangle
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from CAFS import CAFS_V6

def graficar_elipse(pos, score, color_linea, ax=None):
    """
    Traducción de la función MATLAB graficar_elipse.

    Parámetros
    ----------
    pos : array
        Índices de las muestras pertenecientes a una clase.
    score : ndarray
        Matriz de puntuaciones del PCA.
    color_linea : color
        Color de la elipse.
    ax : matplotlib.axes.Axes, opcional
        Ejes donde se dibujará la elipse.
    """

    if ax is None:
        ax = plt.gca()

    # Seleccionar los scores PC1 y PC2
    pts = score[pos, :2]

    # Calcular media y matriz de covarianza
    mu = np.mean(pts, axis=0)
    Sigma = np.cov(pts, rowvar=False)

    # Nivel de probabilidad del 95 %
    p = 0.95
    chi2val = chi2.ppf(p, df=2)

    # Descomposición en autovalores y autovectores
    D, V = np.linalg.eigh(Sigma)

    # Calcular radios de la elipse
    radii = np.sqrt(np.maximum(D, 0) * chi2val)

    # Parametrizar la elipse
    theta = np.linspace(0, 2 * np.pi, 200)

    ellipse_unit = np.vstack((
        np.cos(theta),
        np.sin(theta)
    ))

    # Transformar y trasladar la elipse
    ellipse_coords = (
        V @ np.diag(radii) @ ellipse_unit
    )

    ellipse_coords = ellipse_coords + mu[:, np.newaxis]

    # Dibujar elipse
    ax.plot(
        ellipse_coords[0, :],
        ellipse_coords[1, :],
        color=color_linea,
        linewidth=1.5
    )


def des_clasifH(X, Y, clasif, N, K, random_state=None):
    """Devuelve (Y_pred, Model).

    Y_pred tiene forma (n_test, 2, N): etiquetas reales y predichas.
    Model es el clasificador entrenado en la última iteración.
    K es la proporción de prueba (0 < K < 1).

    Se emplea partición estratificada si es factible. Para mantener el
    tamaño de prueba constante, train_test_split usa ceil(K*n_muestras).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()
    if X.ndim != 2 or len(Y) != X.shape[0]:
        raise ValueError("X debe ser 2D e Y debe tener una etiqueta por fila de X.")
    if not (0 < K < 1):
        raise ValueError("K debe estar entre 0 y 1.")
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N debe ser un entero positivo.")
    if clasif not in (1, 2, 3):
        raise ValueError("clasif debe ser 1 (SVM), 2 (lineal) o 3 (ANN).")
    if not np.isfinite(X).all():
        raise ValueError("X contiene valores NaN o infinitos.")

    classes, counts = np.unique(Y, return_counts=True)
    n_test = int(np.ceil(K * len(Y)))
    n_train = len(Y) - n_test
    if counts.min() < 2 or min(n_test, n_train) < len(classes):
        raise ValueError("No es posible estratificar: revise K y el tamaño de las clases.")

    rng = np.random.default_rng(random_state)
    Y_pred = np.empty((n_test, 2, N), dtype=Y.dtype)
    Model = None
    for i in range(N):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        XTrain, XTest, YTrain, YTest = train_test_split(
            X, Y, test_size=K, stratify=Y, random_state=seed
        )
        if clasif == 1:
            # Aproximación al ECOC multiclase con SVM estandarizada.
            Model = make_pipeline(StandardScaler(), OneVsOneClassifier(SVC(kernel="rbf")))
        elif clasif == 2:
            # Aproximación lineal: no es idéntica a templateLinear de MATLAB.
            #Model = OneVsOneClassifier(LinearSVC(max_iter=10000))
            Model = LinearSVC(dual=True, max_iter=10000, random_state=42)
        else:
            hidden = max(1, int(np.floor(np.sqrt(X.shape[1]) + 0.5)))
            Model = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=1000,
                              random_state=seed)
            )
        Model.fit(XTrain, YTrain)
        Y_pred[:, 0, i] = YTest
        Y_pred[:, 1, i] = Model.predict(XTest)
        print(f"iteration_{i + 1}_de_{N}")
    return Y_pred, Model



def GenerarCMHeatmapH(Y_pred, variedades, clasif, full_opt, clases=None, carpeta_figuras="Figuras"):
    """Devuelve CM con forma (n_clases, n_clases, n_repeticiones).

    Y_pred: (n_prueba, 2, n_repeticiones), columnas [real, predicha].
    variedades: nombres de clases ordenados como `clases`.
    clases: etiquetas originales en el orden deseado; si se omite, se
            utiliza el orden ascendente de etiquetas presentes en Y_pred.
    """
    Y_pred = np.asarray(Y_pred)
    if Y_pred.ndim != 3 or Y_pred.shape[1] != 2:
        raise ValueError("Y_pred debe tener dimensiones (n_prueba, 2, N).")
    if Y_pred.shape[2] == 0:
        raise ValueError("Y_pred debe contener al menos una repetición.")

    nombres = list(variedades)
    if clases is None:
        clases = np.unique(Y_pred[:, :, :].ravel())
    clases = np.asarray(clases).ravel()
    if len(nombres) != len(clases):
        raise ValueError("El número de nombres en variedades debe coincidir con el de clases.")

    matrices = []
    for i in range(Y_pred.shape[2]):
        y_real = Y_pred[:, 0, i]
        y_estimado = Y_pred[:, 1, i]
        matrices.append(confusion_matrix(y_real, y_estimado, labels=clases))
    CM = np.stack(matrices, axis=2)
    CM_mean = np.mean(CM, axis=2)
    sumas = CM_mean.sum(axis=1, keepdims=True)
    CM_norm = np.divide(100.0 * CM_mean, sumas,
                        out=np.zeros_like(CM_mean, dtype=float), where=sumas > 0)

    # MATLAB usa parula; viridis es una alternativa disponible en Matplotlib.
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(CM_norm, cmap='viridis', vmin=0, vmax=100,
                   interpolation='nearest', aspect='equal')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Accuracy (%)', fontsize=14)
    cb.ax.tick_params(labelsize=11)

    n = len(clases)
    ax.set_xticks(np.arange(n), labels=nombres, rotation=45, ha='right')
    ax.set_yticks(np.arange(n), labels=nombres)
    ax.set_xlabel('Predicted class', fontsize=14)
    ax.set_ylabel('Real class', fontsize=14)
    ax.tick_params(axis='both', labelsize=11)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, f'{CM_norm[i, j]:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')
            elif CM_norm[i, j] > 0:
                ax.text(j, i, f'{CM_norm[i, j]:.1f}', ha='center', va='center',
                        fontsize=10, color='white')

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.tight_layout()

    nombres_modelos = {1: 'SVM', 2: 'LDA', 3: 'ANN'}
    if clasif not in nombres_modelos:
        raise ValueError('clasif debe ser 1 (SVM), 2 (LDA) o 3 (ANN).')
    nombre_modelo = nombres_modelos[clasif]
    destino = Path(carpeta_figuras)
    destino.mkdir(parents=True, exist_ok=True)
    nombre = f'{nombre_modelo}_CM_{full_opt}'
    fig.savefig(destino / f'{nombre}.png', dpi=600, bbox_inches='tight')
    fig.savefig(destino / f'{nombre}.eps', format='eps', bbox_inches='tight')
    plt.show()
    return CM


def metricas_CMH(Y, CM, full_opt, clasif, variedades=None,
                 decimales=3):
    """
    Calcula métricas por clase y métricas globales a partir
    de matrices de confusión de N repeticiones hold-out.

    CM: ndarray de dimensiones (n_clases, n_clases, N).

    Se asume:
        Filas = clases reales.
        Columnas = clases predichas.

    Retorno:
        Si variedades no es None:
            DataFrame con métricas por clase.

        Si variedades es None:
            ndarray con mínimo, máximo y desviación
            estándar del F1 por clase.

    Las métricas globales se muestran en consola y se
    guardan en una hoja adicional de metrics.xlsx.
    """

    # ==================================================
    # 1. VALIDACIÓN DE ENTRADAS
    # ==================================================

    Y = np.asarray(Y).ravel()
    CM = np.asarray(CM, dtype=float)

    if CM.ndim == 2:
        CM = CM[:, :, np.newaxis]

    if CM.ndim != 3 or CM.shape[0] != CM.shape[1]:
        raise ValueError(
            "CM debe tener dimensiones "
            "(n_clases, n_clases, N)."
        )

    n_clases, _, N = CM.shape

    if N < 1:
        raise ValueError(
            "Debe existir al menos una matriz de confusión."
        )

    if len(np.unique(Y)) != n_clases:
        raise ValueError(
            "El número de clases de Y no coincide con CM."
        )

    if not np.isfinite(CM).all() or np.any(CM < 0):
        raise ValueError(
            "CM contiene valores negativos, NaN o infinitos."
        )

    if not np.allclose(CM, np.round(CM)):
        raise ValueError(
            "Las matrices de confusión deben contener conteos."
        )

    # ==================================================
    # 2. INICIALIZACIÓN DE MÉTRICAS POR CLASE
    # ==================================================

    accuracy = np.zeros((n_clases, N))
    precision = np.zeros((n_clases, N))
    recall = np.zeros((n_clases, N))
    F1 = np.zeros((n_clases, N))

    # ==================================================
    # 3. INICIALIZACIÓN DE MÉTRICAS GLOBALES
    # ==================================================

    globales = {
        "Accuracy global": np.zeros(N),
        "Error global": np.zeros(N),
        "Balanced Accuracy": np.zeros(N),
        "Precision macro": np.zeros(N),
        "Recall macro": np.zeros(N),
        "F1 macro": np.zeros(N),
        "F1 weighted": np.zeros(N),
        "F1 micro": np.zeros(N)
    }

    # ==================================================
    # 4. FUNCIÓN DE DIVISIÓN SEGURA
    # ==================================================

    def division_segura(numerador, denominador):
        numerador = np.asarray(
            numerador, dtype=float
        )
        denominador = np.asarray(
            denominador, dtype=float
        )

        return np.divide(
            numerador,
            denominador,
            out=np.zeros_like(
                numerador, dtype=float
            ),
            where=denominador != 0
        )

    # ==================================================
    # 5. MÉTRICAS POR REPETICIÓN HOLD-OUT
    # ==================================================

    for i in range(N):

        C = CM[:, :, i]

        total = np.sum(C)

        if total <= 0:
            raise ValueError(
                f"La matriz de confusión {i + 1} está vacía."
            )

        TP = np.diag(C)
        FP = np.sum(C, axis=0) - TP
        FN = np.sum(C, axis=1) - TP
        TN = total - (TP + FP + FN)

        # ----------------------------------------------
        # 5.1. Métricas individuales por clase
        # ----------------------------------------------

        # Accuracy One-vs-Rest
        accuracy[:, i] = division_segura(
            TP + TN,
            np.full(n_clases, total)
        )

        # Precision
        precision[:, i] = division_segura(
            TP,
            TP + FP
        )

        # Recall
        recall[:, i] = division_segura(
            TP,
            TP + FN
        )

        # F1-score
        F1[:, i] = division_segura(
            2 * precision[:, i] * recall[:, i],
            precision[:, i] + recall[:, i]
        )

        # ----------------------------------------------
        # 5.2. Métricas globales
        # ----------------------------------------------

        globales["Accuracy global"][i] = (
            np.sum(TP) / total
        )

        globales["Error global"][i] = (
            1 - globales["Accuracy global"][i]
        )

        # Macro Recall = Balanced Accuracy
        # cuando todas las clases tienen soporte.
        globales["Balanced Accuracy"][i] = (
            np.mean(recall[:, i])
        )

        globales["Precision macro"][i] = (
            np.mean(precision[:, i])
        )

        globales["Recall macro"][i] = (
            np.mean(recall[:, i])
        )

        globales["F1 macro"][i] = (
            np.mean(F1[:, i])
        )

        # F1 ponderado por el número real de muestras
        soporte = np.sum(C, axis=1)

        globales["F1 weighted"][i] = (
            np.sum(F1[:, i] * soporte) / total
        )

        # F1 micro
        TP_total = np.sum(TP)
        FP_total = np.sum(FP)
        FN_total = np.sum(FN)

        globales["F1 micro"][i] = float(
            division_segura(
                2 * TP_total,
                2 * TP_total + FP_total + FN_total
            )
        )

    # ==================================================
    # 6. RESUMEN ESTADÍSTICO
    # ==================================================

    def resumen(matriz):

        media = np.mean(matriz, axis=1)

        if N > 1:
            desviacion = np.std(
                matriz,
                axis=1,
                ddof=1
            )
        else:
            desviacion = np.zeros(
                matriz.shape[0]
            )

        return media, desviacion

    accuracy_mean, accuracy_std = resumen(accuracy)
    precision_mean, precision_std = resumen(precision)
    recall_mean, recall_std = resumen(recall)
    F1_mean, F1_std = resumen(F1)

    # Resumen global: media y desviación estándar
    global_mean = {}
    global_std = {}

    for nombre, valores in globales.items():

        global_mean[nombre] = np.mean(valores)

        global_std[nombre] = (
            np.std(valores, ddof=1)
            if N > 1 else 0.0
        )

    # ==================================================
    # 7. TABLA DE MÉTRICAS GLOBALES
    # ==================================================

    tabla_global = pd.DataFrame({
        "Metrica": list(globales.keys()),
        "Media": [
            global_mean[k]
            for k in globales
        ],
        "Desviacion_estandar": [
            global_std[k]
            for k in globales
        ],
        "Minimo": [
            np.min(globales[k])
            for k in globales
        ],
        "Maximo": [
            np.max(globales[k])
            for k in globales
        ]
    })

    # ==================================================
    # 8. IDENTIFICACIÓN DEL CLASIFICADOR
    # ==================================================

    modelos = {
        1: "SVM",
        2: "LDA",
        3: "ANN"
    }

    modelo_name = modelos.get(
        clasif,
        f"Modelo_{clasif}"
    )

    # ==================================================
    # 9. MOSTRAR MÉTRICAS GLOBALES
    # ==================================================

    print("\n" + "=" * 75)
    print(
        f"DESEMPEÑO GLOBAL: {modelo_name}"
    )
    print(
        f"Configuración: {full_opt} | "
        f"Repeticiones hold-out: {N}"
    )
    print(
        "Valores: media ± desviación estándar"
    )
    print("=" * 75)

    for nombre in globales:

        media = global_mean[nombre]
        desviacion = global_std[nombre]

        print(
            f"{nombre:<25}: "
            f"{media:.{decimales}f} ± "
            f"{desviacion:.{decimales}f}"
        )

    print("=" * 75)

    # ==================================================
    # 10. CONSERVAR RETORNO SIN VARIEDADES
    # ==================================================

    if variedades is None:

        return np.column_stack((
            np.min(F1, axis=1),
            np.max(F1, axis=1),
            F1_std
        ))

    # ==================================================
    # 11. TABLA DE MÉTRICAS POR CLASE
    # ==================================================

    variedades = list(variedades)

    if len(variedades) != n_clases:
        raise ValueError(
            "La cantidad de nombres de variedades "
            "debe coincidir con el número de clases."
        )

    result = pd.DataFrame({
        "Clase": variedades,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "F1_mean": F1_mean,
        "F1_std": F1_std
    })

    # ==================================================
    # 12. TABLA FORMATEADA PARA CONSOLA
    # ==================================================

    def formatear(media, desviacion):

        return [
            f"{m:.{decimales}f} ± "
            f"{s:.{decimales}f}"
            for m, s in zip(media, desviacion)
        ]

    tabla_consola = pd.DataFrame({
        "Clase": variedades,
        "Accuracy OvR": formatear(
            accuracy_mean, accuracy_std
        ),
        "Precision": formatear(
            precision_mean, precision_std
        ),
        "Recall": formatear(
            recall_mean, recall_std
        ),
        "F1-score": formatear(
            F1_mean, F1_std
        )
    })

    print("\n" + "=" * 105)
    print(
        f"DESEMPEÑO POR VARIEDAD: {modelo_name}"
    )
    print(
        f"Configuración: {full_opt} | "
        f"Repeticiones: {N}"
    )
    print("=" * 105)

    print(
        tabla_consola.to_string(
            index=False,
            col_space=17
        )
    )

    print("=" * 105 + "\n")

    # ==================================================
    # 13. EXPORTAR RESULTADOS A EXCEL
    # ==================================================

    archivo = Path("metrics.xlsx")

    hoja_clases = f"{modelo_name}_{full_opt}"
    hoja_global = f"{modelo_name}_{full_opt}_Global"

    if archivo.exists():
        modo = "a"
    else:
        modo = "w"

    writer_kwargs = {
        "path": archivo,
        "engine": "openpyxl",
        "mode": modo
    }

    if modo == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(**writer_kwargs) as writer:

        # Hoja de métricas por clase
        result.to_excel(
            writer,
            sheet_name=hoja_clases,
            index=False
        )

        # Hoja de métricas globales
        tabla_global.to_excel(
            writer,
            sheet_name=hoja_global,
            index=False
        )

        # Formato de ambas hojas
        for nombre_hoja in [hoja_clases, hoja_global]:

            ws = writer.sheets[nombre_hoja]

            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions

            # Encabezado verde con letras blancas
            for celda in ws[1]:

                celda.fill = PatternFill(
                    start_color="548235",
                    end_color="548235",
                    fill_type="solid"
                )

                celda.font = Font(
                    bold=True,
                    color="FFFFFF"
                )

                celda.alignment = Alignment(
                    horizontal="center",
                    vertical="center"
                )

            # Formato numérico
            for fila in ws.iter_rows(
                min_row=2
            ):
                for celda in fila:

                    if isinstance(
                        celda.value,
                        (int, float, np.number)
                    ):
                        celda.number_format = (
                            "0." + "0" * decimales
                        )

            # Ajustar ancho de columnas
            for columna in ws.columns:

                letra = get_column_letter(
                    columna[0].column
                )

                ancho = max(
                    len(str(celda.value))
                    if celda.value is not None
                    else 0
                    for celda in columna
                )

                ws.column_dimensions[letra].width = (
                    min(max(ancho + 3, 15), 35)
                )

    # ==================================================
    # 14. MOSTRAR ARCHIVO GENERADO
    # ==================================================

    print(
        f"Resultados guardados en: {archivo.resolve()}"
    )

    print(
        f"Hoja por clase: {hoja_clases}"
    )

    print(
        f"Hoja global: {hoja_global}"
    )

    return result



def Sel2VarFSCNCA(X, Y, LimRec, bandas, nBandas, classes, bandas_rango,
                  j=None, random_state=42, output_dir='Figuras'):
    """
    Devuelve índices base cero de exactamente j bandas,
    o utiliza un umbral cuando j=None.

    Parámetros
    ----------
    X : ndarray
        Matriz espectral (muestras x variables).
    Y : ndarray
        Etiquetas de las clases.
    LimRec : list
        [modo, umbral]. Se ignora para seleccionar
        cuando j está especificado.
    bandas : ndarray
        Longitudes de onda.
    nBandas : ndarray
        Vector original de bandas.
    classes : ndarray
        Clases del conjunto de datos.
    bandas_rango : ndarray
        Intervalo original de bandas.
    j : int, opcional
        Número exacto de variables a seleccionar.
    random_state : int
        Semilla de reproducibilidad.
    output_dir : str
        Directorio de las figuras.

    Retorna
    -------
    pos : ndarray
        Índices Python (base 0) de las variables seleccionadas.
    """

    # --------------------------------------------------
    # 1. PREPARAR Y VALIDAR LOS DATOS
    # --------------------------------------------------

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()
    bandas = np.asarray(bandas, dtype=float).ravel()
    classes = np.asarray(classes).ravel()

    if (
        X.ndim != 2
        or X.shape[0] != Y.size
        or X.shape[1] != bandas.size
    ):
        raise ValueError(
            'Dimensiones incompatibles de X, Y y bandas.'
        )

    if not np.isfinite(X).all():
        raise ValueError('X contiene NaN o infinitos.')

    if len(np.unique(Y)) < 2:
        raise ValueError('NCA requiere al menos dos clases.')

    if j is not None and (
        not isinstance(j, (int, np.integer))
        or isinstance(j, (bool, np.bool_))
        or not 1 <= j <= X.shape[1]
    ):
        raise ValueError(
            f'j debe ser un entero entre 1 y {X.shape[1]}.'
        )

    # --------------------------------------------------
    # 2. CALCULAR RELEVANCIA NCA
    # --------------------------------------------------

    X_scaled = StandardScaler().fit_transform(X)

    n_components = min(
        10,
        X.shape[1],
        X.shape[0] - 1
    )

    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components,
        random_state=random_state,
        max_iter=200,
        tol=1e-5
    )

    nca.fit(X_scaled, Y)

    # Relevancia aproximada de cada variable
    pesos = np.linalg.norm(
        nca.components_,
        axis=0
    )

    # Normalización para visualización
    if pesos.max() > 0:
        pesos = 8 * pesos / pesos.max()

    # --------------------------------------------------
    # 3. SELECCIONAR VARIABLES ESPECTRALES
    # --------------------------------------------------

    if j is not None:

        # Seleccionar las j variables con mayor relevancia
        # y ordenar sus posiciones espectrales.
        pos = np.sort(
            np.argsort(
                -pesos,
                kind='stable'
            )[:j]
        )

        # Umbral visual de referencia
        umbral = float(np.min(pesos[pos]))

    else:

        if LimRec is None or len(LimRec) < 2:
            raise ValueError(
                'Sin j, especifique LimRec=[1, umbral].'
            )

        if int(LimRec[0]) == 0:
            raise NotImplementedError(
                'La búsqueda automática del umbral '
                'no está implementada. '
                'Utilice j o LimRec=[1, umbral].'
            )

        umbral = float(LimRec[1])

        pos = np.flatnonzero(
            pesos >= umbral
        )

        if not pos.size:
            raise ValueError(
                'El umbral no seleccionó ninguna variable.'
            )


    # --------------------------------------------------
    # 3.1. MOSTRAR VARIABLES RELEVANTES SELECCIONADAS
    # --------------------------------------------------

    import pandas as pd

    # Ordenar las variables seleccionadas por relevancia
    orden_relevancia = pos[
        np.argsort(-pesos[pos], kind="stable")
    ]

    # Construir tabla de resultados
    tabla_relevancia = pd.DataFrame({
        "Ranking": np.arange(1, len(pos) + 1),
        "Variable": orden_relevancia + 1,
        "Wavelength (nm)": bandas[orden_relevancia],
        "NCA Weight": pesos[orden_relevancia]
    })

    # Mostrar tabla en consola
    print("\n" + "=" * 65)
    print("FSCNCA: VARIABLES ESPECTRALES SELECCIONADAS")
    print("=" * 65)

    print(
        tabla_relevancia.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("=" * 65)
    print(f"Total de variables seleccionadas: {len(pos)}")
    print("=" * 65)

    # Guardar tabla de relevancia
    carpeta_resultados = Path("Resultados")
    carpeta_resultados.mkdir(
        parents=True,
        exist_ok=True
    )

    tabla_relevancia.to_csv(
        carpeta_resultados / "FSCNCA_Selected_Wavelengths.csv",
        index=False,
        float_format="%.3f"
    )

    tabla_relevancia.to_excel(
        carpeta_resultados / "FSCNCA_Selected_Wavelengths.xlsx",
        index=False
    )

    # --------------------------------------------------
    # 4. PREPARAR DIRECTORIO Y PERFILES MEDIOS
    # --------------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    perfiles = [
        np.mean(X[Y == clase], axis=0)
        for clase in classes
        if np.any(Y == clase)
    ]

    # --------------------------------------------------
    # 5. FIGURA DE RELEVANCIA NCA
    # --------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for perfil in perfiles:
        ax1.plot(
            bandas,
            perfil,
            '-k',
            linewidth=1
        )

    ax1.set_xlabel(
        'Wavelength (nm)',
        fontsize=14
    )

    ax1.set_ylabel(
        'Normalized reflectance',
        fontsize=14
    )

    ax1.set_ylim(-7, 3)
    ax1.set_xlim(bandas.min(), bandas.max())
    ax1.grid(alpha=0.25)

    # Segundo eje: relevancia
    ax2 = ax1.twinx()

    ax2.plot(
        bandas,
        pesos,
        '-g',
        linewidth=1.2
    )

    ax2.axhline(
        umbral,
        color='blue',
        linestyle='--',
        label='Reference threshold'
    )

    ax2.scatter(
        bandas[pos],
        pesos[pos],
        color='red',
        s=25,
        zorder=3
    )

    # Líneas verticales de bandas seleccionadas
    for idx in pos:
        ax2.axvline(
            bandas[idx],
            color='red',
            linestyle='--',
            linewidth=0.7,
            alpha=0.6
        )

    ax2.set_ylabel(
        'Relative relevance (approximate NCA)',
        fontsize=14
    )

    ax2.set_ylim(0, 8)

    fig.tight_layout()

    fig.savefig(
        output_dir / 'Fig_FSCNA.png',
        dpi=600,
        bbox_inches='tight'
    )

    fig.savefig(
        output_dir / 'Fig_FSCNA.eps',
        format='eps',
        bbox_inches='tight'
    )

    plt.show()

    # --------------------------------------------------
    # 6. FIGURA DE BANDAS SELECCIONADAS
    # --------------------------------------------------

    fig2, ax = plt.subplots(figsize=(12, 6))

    for perfil in perfiles:
        ax.plot(
            bandas,
            perfil,
            '-k',
            linewidth=1
        )

    for idx in pos:
        ax.axvline(
            bandas[idx],
            color='red',
            linestyle='--',
            linewidth=0.8
        )

    ax.set_xlim(
        bandas.min(),
        bandas.max()
    )

    ax.set_ylim(-2.5, 3)

    ax.set_xlabel(
        'Wavelength (nm)',
        fontsize=14
    )

    ax.set_ylabel(
        'Normalized reflectance',
        fontsize=14
    )

    ax.grid(alpha=0.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig2.tight_layout()

    fig2.savefig(
        output_dir / 'Fig_FSCNA1.png',
        dpi=600,
        bbox_inches='tight'
    )

    fig2.savefig(
        output_dir / 'Fig_FSCNA1.eps',
        format='eps',
        bbox_inches='tight'
    )

    plt.show()

    # --------------------------------------------------
    # 7. MOSTRAR RESULTADOS
    # --------------------------------------------------

    print(
        f'FSCNCA aproximado: {len(pos)} '
        f'variables seleccionadas; '
        f'índices Python: {pos}'
    )

    print(
        'Longitudes de onda:',
        bandas[pos]
    )

    return pos




def sel2varSFSS(X, Y, k, bandas, nBandas, bandas_rango, random_state=42):

    """
    Selección secuencial hacia adelante (SFS)
    utilizando LDA y validación cruzada de 5 pliegues.

    Parámetros
    ----------
    X : ndarray
        Matriz de predictores (n_muestras, n_variables).
    Y : ndarray
        Etiquetas de las clases.
    k : int
        Número máximo de variables a seleccionar.
    bandas : ndarray
        Longitudes de onda correspondientes a X.
    nBandas : ndarray
        Vector completo de longitudes de onda.
    bandas_rango : array
        Límites originales de bandas en MATLAB (base 1).
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    pos : ndarray
        Índices Python (base 0) seleccionados.
    Acc : ndarray
        Accuracy promedio de validación cruzada
        en cada iteración.
    """

    # PREPARAR LOS DATOS

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()

    bandas = np.asarray(bandas).ravel()
    nBandas = np.asarray(nBandas).ravel()
    bandas_rango = np.asarray(bandas_rango).ravel()

    obs, nFeatures = X.shape

    if len(Y) != obs:
        raise ValueError(
            "El número de etiquetas no coincide "
            "con las observaciones de X."
        )

    if len(bandas) != nFeatures:
        raise ValueError(
            "El número de longitudes de onda debe "
            "coincidir con las columnas de X."
        )

    if k < 1 or k > nFeatures:
        raise ValueError(
            "k debe estar entre 1 y el número "
            "de variables disponibles."
        )

    # INICIALIZAR VARIABLES DE SELECCIÓN

    sel = np.zeros(nFeatures, dtype=bool)
    remaining = np.ones(nFeatures, dtype=bool)
    selIdx = np.array([], dtype=int)
    Acc = np.zeros(k, dtype=float)

    # VALIDACIÓN CRUZADA DE CINCO PLIEGUES
    cvp = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(cvp.split(X, Y))

    # SELECCIÓN SECUENCIAL HACIA ADELANTE

    for i in range(k):
        bestAcc = -np.inf
        bestFeat = None

        # Evaluar cada característica restante
        for f in np.flatnonzero(remaining):

            # Características seleccionadas + candidata
            cand = np.flatnonzero(sel | (np.arange(nFeatures) == f))
            accs = np.zeros(len(splits))

            # Evaluar mediante validación cruzada
            for t, (tr, te) in enumerate(splits):
                try:
                    mdl = LinearDiscriminantAnalysis()
                    mdl.fit(X[np.ix_(tr, cand)], Y[tr])
                    pred = mdl.predict(X[np.ix_(te, cand)])
                    accs[t] = accuracy_score(Y[te],pred)

                except (ValueError, np.linalg.LinAlgError):
                    # Mantener el comportamiento MATLAB
                    accs[t] = 0.0

            meanAcc = np.mean(accs)

            if meanAcc > bestAcc:
                bestAcc = meanAcc
                bestFeat = f

        # Detener si no se encontró una variable válida
        if bestFeat is None:
            break

        # Actualizar selección
        sel[bestFeat] = True
        remaining[bestFeat] = False

        selIdx = np.flatnonzero(sel)
        Acc[i] = bestAcc

        print(
            f"Iter {i+1}: añadida característica "
            f"{bestFeat+1}. "
            f"Acc CV estimada = {bestAcc:.3f}"
        )

    #  SELECCIÓN ALTERNATIVA POR VARIANZA

    if len(selIdx) == 0:

        print(
            "Advertencia: SFS no seleccionó "
            "características. Se seleccionará "
            "la variable de mayor varianza."
        )

        # ddof=1 equivale a var(X,0,1) de MATLAB
        varianzas = np.var(X, axis=0, ddof=1)

        selIdx = np.array([np.argmax(varianzas)])

    print(
        f"\nSeleccionadas {len(selIdx)} de "
        f"{nFeatures} características mediante SFS."
    )

    pos = selIdx

    # GRAFICAR PERFILES ESPECTRALES

    fig, ax = plt.subplots(figsize=(12, 6))
    classes = np.unique(Y)

    # Graficar perfiles medios por clase
    for clase in classes:
        pos_clases = np.flatnonzero(Y == clase)
        profiles_mean = np.mean(X[pos_clases, :], axis=0)
        ax.plot(bandas, profiles_mean, "-k", linewidth=1.0)

    # Graficar bandas seleccionadas
    for indice in pos:
        ax.plot([bandas[indice], bandas[indice]], [-2.5, 3], "--r", linewidth=1.0 )

    #  CONFIGURACIÓN DE LA FIGURA

    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Convertir índices MATLAB a Python
    inicio = int(bandas_rango[0]) - 1
    fin = int(bandas_rango[1]) - 1

    ax.set_xlim(nBandas[inicio], nBandas[fin])
    ax.set_ylim(-2.5, 3)
    ax.set_ylabel("Normalized reflectance", fontsize=14)
    ax.set_xlabel("Wavelength (nm)", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()

    # GUARDAR FIGURAS

    try:
        from rutas import FIG
        carpeta_figuras = Path(FIG)
    except ImportError:
        carpeta_figuras = Path.cwd() / "Figuras"

    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    fig.savefig(carpeta_figuras / "Fig_SFSS.png", dpi=600, bbox_inches="tight" )
    fig.savefig(carpeta_figuras / "Fig_SFSS.eps", format="eps", bbox_inches="tight")

    plt.show()

    return pos, Acc



def sel2varCAFS(X, y, wavelengths, j=None, output_dir='CAFS'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    wavelengths = np.asarray(wavelengths).ravel()
    if X.ndim != 2 or len(y) != X.shape[0] or len(wavelengths) != X.shape[1]:
        raise ValueError('Dimensiones incompatibles entre X, y y wavelengths')
    if j is not None:
        if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 1 <= j <= X.shape[1]:
            raise ValueError('j debe ser un entero entre 1 y el número de bandas')
    exactas = 0 if j is None else int(j)
    objetivo = 10 if j is None else int(j)

    opts = dict(Patience=3, MinImprovement=1e-4, MinFeatures=objetivo,
                MaxIterations=50, UseHPO=True, RandomSeed=42,
                Preprocessing='snv', RegionWidth=25,
                TargetFeatures=objetivo, ExactFeatures=exactas,
                CompactnessPenalty=0.10, ExcessPenaltyPower=2.0,
                AdaptivePenaltyAlpha=2.0)
    results = CAFS_V6(
        X, y, wavelengths, modelName='nb', outerFolds=5,
        innerFolds=5, hpoFolds=3, options=opts, output_dir=output_dir)
    print('\nResumen de validación externa:')
    print(results['summaryTable'].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nEstabilidad de bandas (20 primeras):')
    print(results['featureStabilityTable'].head(20).to_string(index=False,
            float_format=lambda v: f'{v:.3f}'))

    # Como en MATLAB, se utiliza la selección del primer pliegue externo.
    pos = np.sort(np.asarray(results['foldResults'][0]['selectedFeatureIndices'], dtype=int))
    if j is not None and len(pos) != j:
        raise RuntimeError(f'Se esperaban {j} variables y se obtuvieron {len(pos)}')
    print(f'\nBandas seleccionadas ({len(pos)}): {np.array2string(wavelengths[pos], precision=3)}')

    fig, ax = plt.subplots(figsize=(12, 6))
    for clase in np.unique(y):
        ax.plot(wavelengths, X[y == clase].mean(axis=0), '-k', linewidth=1)
    for idx in pos:
        ax.axvline(wavelengths[idx], color='red', linestyle='--', linewidth=1)
    ax.set(xlim=(wavelengths.min(), wavelengths.max()), ylim=(-2.5, 3),
           xlabel='Wavelength (nm)', ylabel='Normalized reflectance')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    out = Path('Figuras')
    out.mkdir(exist_ok=True, parents=True)
    fig.savefig(out / 'Fig_CAFS.png', dpi=600, bbox_inches='tight')
    fig.savefig(out / 'Fig_CAFS.eps', bbox_inches='tight')
    plt.show()
    return pos
