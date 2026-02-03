# Estrategia de Modelado Predictivo: Insurance Claims

**Rol:** Data Scientist Senior | Allianz
**Metodología:** CRISP-DM

* Este documento prersenta un resumen del contenido de la presentación.

## 📋 Resumen Ejecutivo

El objetivo de este proyecto fue desarrollar un modelo predictivo capaz de identificar pólizas de seguros con alta probabilidad de siniestro para optimizar la estrategia de primas. Se aplicó un enfoque riguroso bajo la metodología **CRISP-DM**.

Tras la fase de evaluación, la recomendación técnica es **DETENER el pase a producción**. La evidencia estadística demuestra que los datos actuales carecen de la señal predictiva necesaria, resultando en una operatividad inviable con una tasa de Falsos Positivos de 9:1 en el segmento de mayor riesgo.

## 🎯 Objetivos y KPIs de Negocio

El propósito central es la optimización de primas mediante la detección temprana de riesgo. Para medir el éxito no en términos abstractos (como *accuracy*), sino en impacto operativo, se definieron los siguientes KPIs centrados en el **Top 20%** de las probabilidades generadas (donde se concentraría la acción de negocio):

| KPI | Definición Técnica | Objetivo de Negocio |
| :--- | :--- | :--- |
| **Lift** | Acumulado de clasificación en el Top 20%. | Medir cuánto mejor es el modelo frente al azar. |
| **Precision @ k** | Tasa de acierto en el top 20% de probabilidades. | Minimizar el costo operativo de investigar alertas falsas. |
| **Recall @ k** | % de siniestros detectados en el top 20% del total. | Maximizar la captura de riesgo real. |

## 🔍 Análisis Exploratorio de Datos (EDA)

Se realizó un análisis univariado y bivariado, incluyendo matrices de correlación Phik (para capturar relaciones no lineales).

### Insights Críticos
1.  **Poder Explicativo Débil:** Existe una superposición significativa en las distribuciones de variables entre las clases `claim` (1) y `no-claim` (0). No hay separación lineal evidente.
2.  **Correlaciones Bajas:** Las especificaciones mecánicas del vehículo por sí solas presentan correlaciones cercanas a cero con la variable objetivo. Esto sugiere que el "riesgo" no es intrínseco al vehículo en este dataset, sino probablemente comportamental y demografico.

## ⚙️ Estrategia de Modelado y Selección

Dada la naturaleza desbalanceada del dataset, se evaluaron arquitecturas de ensamble robustas frente al desbalance de clases.

**Modelos Evaluados:**
*   GradientBoostingClassifier
*   EasyEnsembleClassifier
*   BalancedRandomForest

**Resultados de Entrenamiento:**

| Modelo | ROC AUC (Mean) | Recall (Mean) | F1 Score (Mean) |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting** | 0.641 | 0.000 | 0.000 |
| **EasyEnsemble** | **0.630** | **0.715** | **0.152** |
| **BalancedRandom** | 0.601 | 0.340 | 0.138 |

*Nota: Gradient Boosting falló en capturar la clase minoritaria (Recall 0), sesgándose a la clase mayoritaria.*

## 📉 Evaluación de Impacto Operativo

Se seleccionó el **EasyEnsembleClassifier** por su capacidad de recuperación (Recall). Sin embargo, al trasladar las métricas técnicas a métricas de negocio en el Top 20% de riesgo, el modelo es inoperable.

### Matriz de Confusión (Top 20% Riesgo)

| | **Realidad: Siniestro (1)** | **Realidad: No Siniestro (0)** | **Total Predicho** |
| :--- | :---: | :---: | :---: |
| **Predicho: Siniestro (1)**<br>*(Top 20% Riesgo)* | **1,164**<br>*(True Positives)* | **10,555**<br>*(False Positives)* | **11,719**<br>*(Volumen de Alertas)* |
| **Predicho: No Siniestro (0)**<br>*(Resto de la Base)* | **2,584**<br>*(False Negatives)* | **44,289**<br>*(True Negatives)* | **46,873** |
| **Total Real** | **3,748** | **54,844** | **58,592** |

*   **Precision @ k:** 9.9%
*   **Ratio de Ruido:** ~9:1

**Interpretación:** Por cada siniestro real identificado correctamente, el equipo operativo tendría que investigar o incomodar inútilmente a **9 clientes legítimos**. Esto genera una fricción inaceptable con el cliente y riesgo de *churn*.

## 🚀 Conclusiones y Recomendaciones

La aplicación de la metodología CRISP-DM cumplió su función crítica: evitar un despliegue fallido protegiendo el ROI del área.

1.  **Decisión:** **NO IMPLEMENTAR**. El AUC de 0.63 es insuficiente para una operación automatizada.
2.  **Limitación de Datos:** El dataset actual no contiene los *drivers* fundamentales del riesgo.
3.  **Siguientes Pasos:** Redirigir esfuerzos hacia el **enriquecimiento de datos**. Es imperativo integrar fuentes externas (historial de siniestralidad del conductor, telemática, variables demográficas) para mejorar la separabilidad de las clases antes de iterar nuevos modelos.

---
*Repositorio mantenido por Alexis Abreu garzón - Senior Data Scientist*
