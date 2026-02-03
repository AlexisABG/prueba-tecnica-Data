# Conjunto de funciones que sirven como herramientas para el desarrollo de este proyecto

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def extraer_motor_metricas(df, col_name, new_metric_name, new_rpm_name):
    """
    Extrae métrica principal y RPM de columnas tipo '250Nm@2750rpm' de forma eficiente.
    Args:
        df: DataFrame.
        col_name: Nombre de la columna sucia (ej: 'max_torque').
        new_metric_name: Nombre para la magnitud (ej: 'torque_nm').
        new_rpm_name: Nombre para las RPM (ej: 'torque_rpm').
    """
    # 1. Regex compilado para velocidad (captura float + unidad + float + rpm)
    # Soporta Nm, bhp, ps, kw. Ignora espacios y mayúsculas.
    patron = re.compile(r'(\d+\.?\d*)\s*(?:nm|bhp|ps|kw).*?@\s*(\d+\.?\d*)', re.IGNORECASE)

    # 2. Extracción vectorizada usando str.extract (mucho más rápido que .apply fila por fila)
    extracted = df[col_name].str.extract(patron)

    # 3. Asignación directa convirtiendo a float
    df[new_metric_name] = extracted[0].astype(float)
    df[new_rpm_name] = extracted[1].astype(float)

    df.drop(columns=[col_name], inplace=True)

    return df

#############################################

def get_encoding(df, column):
    """
    Aplica One-Hot Encoding a 'region_code' para 22 categorías.
    Genera columnas booleanas (0/1) tipo 'region_C8', 'region_C2'.
    """
    # get_dummies genera las columnas binarias
    # prefix='reg' para nombres cortos: reg_C8, reg_C2...
    # dtype=int para tener 1/0 en lugar de True/False (mejor compatibilidad)
    dummies = pd.get_dummies(df[column], prefix=column, dtype=int)

    # Concatenamos horizontalmente y eliminamos la original
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[column])

    return df

########################################################


def binarizar_columnas(df, columnas):
    """
    Transforma columnas 'Yes'/'No' (insensible a mayúsculas) a 1/0.
    Args:
        df: DataFrame.
        columnas: String (una columna) o Lista de strings (múltiples columnas).
    """
    # 1. Normalizar input a lista para procesar uniforme
    if isinstance(columnas, str):
        columnas = [columnas]

    # 2. Diccionario de mapeo optimizado
    # Cubrimos variaciones comunes para evitar hacer .str.lower() costoso si no es necesario,
    # pero para máxima seguridad aplicaremos normalización primero.
    mapeo = {'yes': 1, 'no': 0}

    for col in columnas:
        if col in df.columns:
            # Normalizamos a minúsculas, mapeamos y rellenamos NaNs con 0 (o np.nan según prefiera)
            # Usamos map() que es más rápido que apply()
            df[col] = df[col].astype(str).str.lower().map(mapeo)

            # Opcional: Manejo de valores no esperados (ej. nulos se vuelven NaN)
            # Si quiere forzar 0 para todo lo que no sea yes:
            # df[col] = df[col].fillna(0).astype(int)


    return df

#####################################################################################

def calcular_mi_bootstrap_profundo(X, y, n_iteraciones=50):
    """
    Ejecuta un Bootstrapping profundo para estabilizar el ranking de importancia.
    Recomendado para N > 50k filas.

    Args:
        n_iteraciones: 50 o 100 recomendado para convergencia estadística.
    """

    # Máscara de discretas (fija)
    # Importante: Asegurarse que X tenga dtypes correctos antes de entrar aquí
    discrete_mask = (X.dtypes == 'int64') | (X.dtypes == 'int32') | (X.dtypes == 'uint8') | (X.dtypes == 'bool')

    # Separar clases
    df_full = pd.concat([X, y], axis=1)
    majority = df_full[df_full['claim_status'] == 0]
    minority = df_full[df_full['claim_status'] == 1]

    n_minority = len(minority)
    accumulated_scores = []

    print(f"--- Iniciando Bootstrapping Profundo ({n_iteraciones} ciclos) ---")
    print(f"Clase Minoritaria: {n_minority} casos por iteración.")
    print(f"Total procesado por ciclo: {n_minority * 2} filas.")

    # Ciclo con barra de progreso
    for i in tqdm(range(n_iteraciones), desc="Calculando MI"):
        # Resample
        majority_downsampled = resample(
            majority,
            replace=False,
            n_samples=n_minority,
            random_state=i # Variabilidad controlada
        )

        # Dataset balanceado temporal
        df_iter = pd.concat([majority_downsampled, minority])
        y_iter = df_iter.pop('claim_status')
        X_iter = df_iter

        # Cálculo MI
        scores = mutual_info_classif(
            X_iter,
            y_iter,
            discrete_features=discrete_mask,
            random_state=42,
            n_neighbors=3
        )
        accumulated_scores.append(scores)

    # Convertir lista de arrays a DataFrame
    # Filas: Iteraciones, Columnas: Features
    scores_matrix = pd.DataFrame(accumulated_scores, columns=X.columns)

    # Estadística Robusta
    results = pd.DataFrame({
        'MI_Mean': scores_matrix.mean(),
        'MI_Median': scores_matrix.median(), # La mediana es más robusta a outliers de muestreo
        'MI_Std': scores_matrix.std(),
        'CI_Lower': scores_matrix.quantile(0.025), # Intervalo Confianza 95% (Cota inferior)
        'CI_Upper': scores_matrix.quantile(0.975)  # Intervalo Confianza 95% (Cota superior)
    })

    # Ordenar por Mediana (más seguro que media en distribuciones sesgadas)
    results = results.sort_values(by='MI_Median', ascending=False)

    # --- Visualización Profesional ---
    plt.figure(figsize=(12, 10))

    # Graficamos los Top 20
    top_20 = results.head(20)

    # Gráfico de puntos con barras de error (Intervalo de Confianza)
    y_pos = np.arange(len(top_20))

    plt.errorbar(
        x=top_20['MI_Median'],
        y=y_pos,
        xerr=[top_20['MI_Median'] - top_20['CI_Lower'], top_20['CI_Upper'] - top_20['MI_Median']],
        fmt='o', color='darkblue', ecolor='skyblue', capsize=5, label='Mediana con IC 95%'
    )

    plt.yticks(y_pos, top_20.index)
    plt.gca().invert_yaxis()
    plt.title(f'Ranking de Importancia Robusta (N={n_iteraciones})\nMediana e Intervalo de Confianza 95%')
    plt.xlabel('Información Mutua (Bits)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results



####################################################

def calcular_importancia_brf(X, y):
    """
    Calcula Feature Importance usando Balanced Random Forest de imblearn.
    Este modelo realiza undersampling automático en cada árbol, ideal para
    datasets desbalanceados.
    """
    print(f"Entrenando Balanced Random Forest sobre {X.shape}...")

    # Configuración robusta
    # n_estimators=200: Suficientes árboles para estabilizar la varianza
    # sampling_strategy='auto': Balancea 50/50 en cada bootstrap
    # replacement=False: Muestreo sin reemplazo (más estricto)
    brf = BalancedRandomForestClassifier(
        n_estimators=200,
        random_state=42,
        sampling_strategy='auto',
        replacement=False,
        n_jobs=-1 # Paralelismo
    )

    brf.fit(X, y)

    # Extraer importancias
    importances = brf.feature_importances_

    # Crear DataFrame
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Visualización
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feature_imp.head(20),
        y='Feature',
        x='Importance',
        palette='magma' # Paleta distinta para diferenciar visualmente del análisis anterior
    )
    plt.title('Top 20 Drivers: Balanced Random Forest (Imblearn)')
    plt.xlabel('Importancia (Gini Decrease Mean)')
    plt.tight_layout()
    plt.show()

    return feature_imp


######################################################################################

def calcular_permutacion_segura(model, X, y, n_repeats=5):
    """
    Calcula Permutation Importance con barra de progreso.
    Retorna SOLO el DataFrame (sin graficar) para evitar errores visuales.
    """
    # 1. Asegurar que el modelo esté entrenado
    print("Verificando modelo...")
    try:
        model.predict_proba(X.iloc[:5])
    except:
        print("Entrenando modelo (BalancedRandomForest)...")
        model.fit(X, y)

    # 2. Score base
    print("Calculando línea base...")
    # Usamos values para evitar problemas de índices
    y_true = y.values if hasattr(y, 'values') else y
    y_pred_base = model.predict_proba(X)[:, 1]
    baseline_score = roc_auc_score(y_true, y_pred_base)
    print(f"Baseline ROC-AUC: {baseline_score:.4f}")

    importances_mean = []
    importances_std = []

    # 3. Iteración visible
    features = X.columns.tolist()
    print(f"Iniciando evaluación de {len(features)} variables...")

    for col in tqdm(features):
        scores_col = []
        # Copia segura de la columna original
        original_col = X[col].values.copy()

        for _ in range(n_repeats):
            # Permutar columna (in-place en el DataFrame temporal X)
            X[col] = np.random.permutation(original_col)

            # Predecir
            y_pred = model.predict_proba(X)[:, 1]
            score = roc_auc_score(y_true, y_pred)

            # Importancia = Base - Nuevo (cuánto cayó)
            scores_col.append(baseline_score - score)

        # Restaurar columna original INMEDIATAMENTE
        X[col] = original_col

        # Guardar estadísticas
        importances_mean.append(np.mean(scores_col))
        importances_std.append(np.std(scores_col))

    # 4. Construir DataFrame final
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importance_Mean': importances_mean,
        'Importance_Std': importances_std
    })

    return df_imp.sort_values(by='Importance_Mean', ascending=False)

  ########################################################################################################


def consolidar_importancias(df, prefixes):
    """
    Agrupa las importancias de las variables dummy en su variable padre.
    """
    df_cons = df.copy()
    
    # Columna temporal para agrupar
    df_cons['Original_Feature'] = df_cons['Feature']
    
    # Barrer prefijos y renombrar
    for prefix in prefixes:
        # Busca filas que empiezan con el prefijo
        mask = df_cons['Feature'].str.startswith(prefix)
        # Asigna el nombre del padre (quitando el _ al final del prefix para limpieza)
        parent_name = prefix.rstrip('_') 
        df_cons.loc[mask, 'Original_Feature'] = parent_name
        
    # Agregación Matemática
    # Media: Se suman las importancias (Aditivo)
    # Std: Se suman las varianzas y se saca raíz (Propagación de error asumiendo independencia)
    # Varianzas = std^2
    
    grouped = df_cons.groupby('Original_Feature').agg(
        Total_Importance=('Importance_Mean', 'sum'),
        Variance_Sum=('Importance_Std', lambda x: np.sum(x**2))
    ).reset_index()
    
    grouped['Combined_Std'] = np.sqrt(grouped['Variance_Sum'])
    
    return grouped[['Original_Feature', 'Total_Importance', 'Combined_Std']].sort_values(by='Total_Importance', ascending=False)


#####################################################################################

def consolidar_importancias(df, prefixes):
    """
    Agrupa las importancias de las variables dummy en su variable padre.
    """
    df_cons = df.copy()
    
    # Columna temporal para agrupar
    df_cons['Original_Feature'] = df_cons['Feature']
    
    # Barrer prefijos y renombrar
    for prefix in prefixes:
        # Busca filas que empiezan con el prefijo
        mask = df_cons['Feature'].str.startswith(prefix)
        # Asigna el nombre del padre (quitando el _ al final del prefix para limpieza)
        parent_name = prefix.rstrip('_') 
        df_cons.loc[mask, 'Original_Feature'] = parent_name
        
    # Agregación Matemática
    # Media: Se suman las importancias (Aditivo)
    # Std: Se suman las varianzas y se saca raíz (Propagación de error asumiendo independencia)
    # Varianzas = std^2
    
    grouped = df_cons.groupby('Original_Feature').agg(
        Total_Importance=('Importance_Mean', 'sum'),
        Variance_Sum=('Importance_Std', lambda x: np.sum(x**2))
    ).reset_index()
    
    grouped['Combined_Std'] = np.sqrt(grouped['Variance_Sum'])
    
    return grouped[['Original_Feature', 'Total_Importance', 'Combined_Std']].sort_values(by='Total_Importance', ascending=False)