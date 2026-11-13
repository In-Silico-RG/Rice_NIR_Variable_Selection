"""
# Seccion 0. Carga de condiciones para el analisis de perfiles de arroz
"""
from pathlib import Path
#import argparse
import numpy as np
#import pandas as pd
from scipy.io import loadmat
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.svm import SVC, LinearSVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from FUNCIONES import graficar_elipse, des_clasifH, GenerarCMHeatmapH, metricas_CMH, Sel2VarFSCNCA, sel2varSFSS, sel2varCAFS


print("\n" + "=" * 65)
print("Fin etapa 0")
print("\n" + "=" * 65)
#%% 
"""
# Seccion 1. Configurar directorios y cargar datos espectrales
""" 

# configurar directorios
AQUI = Path(__file__).resolve().parent
DATOS = AQUI
FIG = AQUI / "Figuras"
FIG.mkdir(parents=True, exist_ok=True)

# cargr datos
datos = loadmat(DATOS / "data" / "profile_totales.mat")
profile_totales = datos["profile_totales"]

X = profile_totales[:, :-1].astype(float)
Y = profile_totales[:, -1]

#Eliminar muestras con valores NaN

# Identificar filas que contienen al menos un NaN
nan_mask = np.isnan(X).any(axis=1)

# Eliminar las mismas filas en X e Y
X = X[~nan_mask, :]
Y = Y[~nan_mask]

print("Muestras eliminadas por NaN:", np.sum(nan_mask))
print("Dimensiones de X:", X.shape)
print("Dimensiones de Y:", Y.shape)


# seleccionar rango de longitudes de onda
datos_bandas = loadmat(DATOS /  "data" / "nBandas.mat")
nBandas = np.asarray(datos_bandas["nBandas"]).ravel()

# Índices originales de MATLAB (1-based)
bandas_rango = [47, 309]

# Conversión a índices de Python (0-based)
inicio = bandas_rango[0] - 1
fin = bandas_rango[1]

bandas = nBandas[inicio:fin]
X = X[:, inicio:fin]

print("\n" + "=" * 65)
print("Número de bandas seleccionadas:", len(bandas))
print("Rango espectral:", bandas[0], "-", bandas[-1], "nm")
print("Fin etapa 1")
print("\n" + "=" * 65)


#%% 
"""
# Seccion 23. Graficar perfiles
""" 
classes = np.unique(Y)

variedades = [ "La Union", "Bellavista", "Mallares", "La Conquista", "Pitipo",
    "Amazonas", "Capirona", "IR43", "La Esperanza", "Tinajones", "La Puntilla",
    "Capoteña"]

if len(classes) != len(variedades):
    raise ValueError("El número de clases no coincide con las 12 variedades.")

# Paleta de colores para las 12 variedades
colors = plt.get_cmap("tab20")(
    np.linspace(0, 1, len(classes))
)


# graficar perfiles espectrales promedio

fig, ax = plt.subplots(figsize=(11, 7))


for i, clase in enumerate(classes):
    pos = Y == clase
    profiles_mean = np.mean(X[pos, :], axis=0)
    ax.plot(
        bandas,
        profiles_mean,
        color=colors[i],
        linewidth=1.5,
        label=variedades[i]
    )

ax.legend(loc="best", fontsize=10)

ax.grid(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(bandas[0], bandas[-1])
ax.set_ylim(-2.5, 2.5)

ax.set_ylabel("Normalized reflectance", fontsize=14)
ax.set_xlabel("Wavelength (nm)", fontsize=14)

ax.tick_params(axis="both", labelsize=14)

fig.tight_layout()
plt.show()

# guardar figura en formato EPS

archivo_salida = FIG / "Fig_prof_means.eps"
fig.savefig(archivo_salida,format="eps", bbox_inches="tight")

plt.show()

print("\n" + "=" * 65)
print("Figura guardada en:", archivo_salida)
print("Fin etapa 2")
print("\n" + "=" * 65)

#%% 
"""
# Seccion 3. Analisis PCA
""" 

# X contiene los perfiles espectrales
# Y contiene las etiquetas de las variedades

X = np.asarray(X, dtype=float)
Y = np.asarray(Y).ravel()

# PCA con centrado automático, sin escalamiento
pca = PCA(n_components=2)

score = pca.fit_transform(X)

# Equivalentes de MATLAB
coeff = pca.components_.T
latent = pca.explained_variance_

# Varianza explicada por componente
varianza = pca.explained_variance_ratio_ * 100

print("Varianza explicada PC1:", varianza[0], "%")
print("Varianza explicada PC2:", varianza[1], "%")
print("Varianza acumulada:", np.sum(varianza), "%")

# graficar los dos primeros componentes
markers = ["o", "s", "D"]

fig, ax = plt.subplots(figsize=(11, 8))

for i, clase in enumerate(classes):

    # Seleccionar muestras de la clase
    pos = Y == clase

    # Repetir marcadores cada tres clases
    marcador = markers[i % len(markers)]

    # Graficar scores
    ax.scatter(score[pos, 0], score[pos, 1], s=40, color=colors[i], marker=marcador,
        label=variedades[i], alpha=0.8)

# Crear leyenda antes de las elipses
ax.legend(loc="best", fontsize=10)

# Graficar elipses de confianza
for i, clase in enumerate(classes):
    pos = np.flatnonzero(Y == clase)
    graficar_elipse(pos, score, colors[i], ax=ax)

# Configurar ejes
ax.set_xlabel("First principal component", fontsize=14)
ax.set_ylabel("Second principal component", fontsize=14)
ax.grid(True, alpha=0.3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", labelsize=14)

fig.tight_layout()

# guardar figuras

# Carpeta de salida
FIG = Path(__file__).resolve().parent / "Figuras"
FIG.mkdir(parents=True, exist_ok=True)

fig.savefig(FIG / "Fig_PCA.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG / "Fig_PCA.eps", format="eps", bbox_inches="tight")

plt.show()

print("\n" + "=" * 65)
print("Figuras PCA guardadas correctamente.")
print("Fin etapa 3")
print("\n" + "=" * 65)


#%% 
"""
# Seccion 4. Modelado full
""" 

# Seleccionar clasificadores
# 1 = SVM, 2 = lineal, 3 = ANN
clasif = [1, 2, 3, 4]
# 1 = espectro completo
# Otros valores corresponden a etapas optimizadas
full_opt = 1
N = 30# Número de repeticiones
K = 0.30# Proporción de datos de prueba (hold-out)

# Ejecutar clasificación
for clasificador in clasif:
    # Entrenar y evaluar el clasificador
    Y_pred, Model = des_clasifH(X, Y, clasificador, N, K)

    # Generar matrices de confusión
    CM = GenerarCMHeatmapH(Y_pred, variedades, clasificador, full_opt)

    # Calcular métricas
    results = metricas_CMH(Y, CM, full_opt, clasificador, variedades)


print("\n" + "=" * 65)
print("Clasificación finalizada.")
print("Fin etapa 4")
print("=" * 65)



#%% 
"""
# Seccion 5. Modelado optimizado
""" 

# Selección de variables espectrales y clasificación
metodo = [1, 2, 3] # Métodos de selección: 1 = FSCNCA, 2 = SFSS, 3 = CAFS
clasif = [1, 2, 3, 4] # Clasificadores: 1 = SVM, 2 = L-SVM, 3 = ANN, 4 = LDA
N = 30 # Número de repeticiones hold-out
K = 0.30 # Proporción de prueba: 30 %


# Archivo de resultados
archivo_excel = Path("metrics.xlsx")

# Almacenar resultados de todos los modelos
resultados_optimizados = {}

# ------------------------------------------------------
# FUNCIÓN AUXILIAR PARA GUARDAR EN EXCEL
# ------------------------------------------------------

def guardar_posiciones_excel(pos, fila_inicio):
    """
    Guarda las posiciones seleccionadas en la hoja
    Consolidado, columna D, sin sobrescribir las
    demás hojas de metrics.xlsx.
    """
    from openpyxl import Workbook, load_workbook

    if archivo_excel.exists():
        wb = load_workbook(archivo_excel)
    else:
        wb = Workbook()

    if "Consolidado" not in wb.sheetnames:
        ws = wb.create_sheet("Consolidado")
    else:
        ws = wb["Consolidado"]

    for j, valor in enumerate(np.asarray(pos).ravel()):
        ws.cell(
            row=fila_inicio + j,
            column=4,
            value=int(valor)
        )

    wb.save(archivo_excel)


# ------------------------------------------------------
# SELECCIÓN DE VARIABLES
# ------------------------------------------------------

for metodo_actual in metodo:

    print("\n" + "=" * 65)
    print(f"MÉTODO DE SELECCIÓN: {metodo_actual}")
    print("=" * 65)

    if metodo_actual == 1:# FSCNCA
        j = 9
        
        LimRec = [1, 3.6]
        
        pos = Sel2VarFSCNCA(X, Y, LimRec, bandas, nBandas, classes, bandas_rango,j=j)

        # Obtener matriz con las j variables seleccionadas
        X_sel = X[:, pos]
        
        print(f"Variables seleccionadas: {len(pos)}")
        print("Longitudes de onda:", bandas[pos])

        guardar_posiciones_excel(pos, 20)
        nombre_metodo = "FSCNCA"

    elif metodo_actual == 2:# SFSS
        k = 9
        pos, Acc = sel2varSFSS(
            X, Y, k, bandas,
            nBandas, bandas_rango
        )

        guardar_posiciones_excel(pos, 37)
        nombre_metodo = "SFSS"

        # Gráfica de accuracy por número de variables
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(np.arange(1, k + 1), np.asarray(Acc).ravel())
        ax.set_xlabel("Number of variables", fontsize=14)
        ax.set_ylabel("Best accuracy", fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(1, k + 1))
        ax.grid(True, axis="y", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()

    elif metodo_actual == 3:# CAFS

        wavelengths = bandas
        y = Y

        j = 9 # numero de ongiyudes seleccionadas
        pos = sel2varCAFS(X, Y, bandas, j=j)

        guardar_posiciones_excel(pos, 54)
        nombre_metodo = "CAFS"

    else:

        raise ValueError(
            f"Método no válido: {metodo_actual}"
        )

    # --------------------------------------------------
    # APLICAR VARIABLES SELECCIONADAS
    # --------------------------------------------------

    pos = np.asarray(pos).ravel().astype(int)

    # IMPORTANTE:
    # Se supone que las funciones de selección
    # devuelven índices Python (base 0).

    X_sel = X[:, pos]

    print(f"\nMétodo: {nombre_metodo}")
    print(f"Variables originales: {X.shape[1]}")
    print(f"Variables seleccionadas: {X_sel.shape[1]}")
    print(f"Posiciones seleccionadas: {pos}")

    # Identificador de configuración
    # 2 = FSCNCA
    # 3 = SFSS
    # 4 = CAFS
    full_opt = metodo_actual + 1

    # --------------------------------------------------
    # CLASIFICACIÓN Y EVALUACIÓN
    # --------------------------------------------------

    resultados_optimizados[nombre_metodo] = {}

    for clasificador in clasif:

        print("\n" + "-" * 65)
        print(
            f"Clasificador: {clasificador} | "
            f"Método: {nombre_metodo}"
        )
        print("-" * 65)

        # Entrenar y evaluar modelo
        Y_pred, Model = des_clasifH(X_sel, Y, clasificador, N, K)

        # Generar matrices de confusión
        CM = GenerarCMHeatmapH(Y_pred, variedades, clasificador, full_opt,
            clases=np.unique(Y)
            )

        # Calcular métricas
        results = metricas_CMH(Y, CM, full_opt, clasificador, variedades,
            decimales=3
            )

        # Guardar resultados en memoria
        resultados_optimizados[nombre_metodo][clasificador] = {
            "pos": pos.copy(),
            "Model": Model,
            "CM": CM,
            "Y_pred": Y_pred,
            "results": results
        }

print("\n" + "=" * 65)
print("Fin etapa 5")
print("=" * 65)
