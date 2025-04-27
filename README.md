# <h1 align="center">Segmentación de clientes para Fashion Retail Co. </h1>

## **Problema de negocio**

- Promociones genéricas con bajo ROI.
- Alta tasa de desuscripción en campañas, reflejando baja fidelidad.

## **Solución propuesta**

Implementar un análisis RFM (Recencia, Frecuencia, Monetario) y clustering K‑Means (k=4) para identificar segmentos clave:

- **Champions** (alto valor)
- **At Risk** (necesitan reenganche)
- **Recent Buyers**, **Frequent Buyers** y **Others** (oportunidades de up‑sell y cross‑sell)

## **Principales hallazgos**

| Segmento | Clientes | Gasto mediano (USD) | ROI simulado |
| --- | --- | --- | --- |
| Champions | 23 | 2 200 | 143 % |
| At Risk | 32 | 1 350 | 667 % |
| Recent Buyers | 48 | 1 800 | 200 % |
| Frequent Buyers | 20 | 1 900 | 180 % |
| Others | 39 | 1 500 | 200 % |

## **Recomendaciones generales**

1. **Automatizar** la segmentación RFM mensual para alimentar campañas personalizadas.
2. **Bundles** de $40–$60 USD con descuentos ligeros para elevar ticket medio.
3. **Programa de fidelidad** y referidos para impulsar frecuencia de compra.
4. **Cupones de re‑enganche** (15 % off) para clientes "At Risk".
5. **Promociones en días valle** (e.g., "Happy Tuesday").

## Fases del proyecto

1. **Limpieza y preparación (Fase 1)**
    - Carga de datos, conversión de fechas, tratamiento de nulos y outliers, normalización de categorías.
2. **EDA (Fase 2)**
    - Estadísticas univariantes, análisis de variables categóricas y tendencias temporales.
3. **Cálculo RFM (Fase 3)**
    - Definición de `snapshot_date`, agregación por cliente, scoring por quintiles, etiquetado de segmentos.
4. **Clustering K‑Means (Fase 4)**
    - Estandarización de features, elección de k (Elbow, Silhouette), entrenamiento y caracterización de clusters.
5. **Visualización (Fase 5)**
    - Scatter R vs F, boxplots de monetary, tamaños de clusters, análisis visual para storytelling.
6. **Recomendaciones y simulación de ROI (Fase 6)**
    - Estrategias por segmento, bundles, fidelidad, re‑enganche, promociones estacionales, cálculo de ROI.

---

---

---

<p align="center">
<big><strong>✨¡Gracias por visitar este repositorio!✨</strong></big>
</p>