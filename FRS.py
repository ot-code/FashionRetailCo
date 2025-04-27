# ================================================================
# 🟢 FASE 1. DATA LOADING AND DATA CLEANING
# ================================================================


# --- Cargar el CSV --- 

from IPython.display import display
import os
import pandas as pd

# --- Obtiene la ruta del directorio donde está el script --- 
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'Fashion_Retail_Sales.csv')

# --- Carga el archivo CSV --- 
df = pd.read_csv(csv_path)

# --- Exploración inicial --- 
display(df.head())        # primeras 5 filas
display(df.tail())        # últimas 5 filas
print(df.shape)           # (nº filas, nº columnas)
print(df.dtypes)          # tipo de cada columna

# --- 2. Conversión de fechas y creación de variables temporales --- 

# Convertir columna de texto a datetime
df['date_purchase'] = pd.to_datetime(df['Date Purchase'], format='%d-%m-%Y')

# Verificar que la conversión a datetime fue exitosa
print(df['date_purchase'].dtype)

# Crear variables auxiliares
df['year']  = df['date_purchase'].dt.year
df['month'] = df['date_purchase'].dt.month
df['day']   = df['date_purchase'].dt.day
df['weekday'] = df['date_purchase'].dt.day_name()

# Verificar cambios
display(df[['Date Purchase','date_purchase','year','month','weekday']].head())

# --- 3. Detección y tratamiento de valores faltantes y outliers --- 

# 3.1 Valores faltantes: conteo de nulos por columna
print(df.isna().sum())

# 3.2 Outliers en “Purchase Amount (USD) 
import numpy as np

# Estadísticas descriptivas
print(df['Purchase Amount (USD)'].describe())

# Identificar outliers usando IQR
Q1 = df['Purchase Amount (USD)'].quantile(0.25)
Q3 = df['Purchase Amount (USD)'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df['Purchase Amount (USD)'] < lower) | (df['Purchase Amount (USD)'] > upper)]
print(f"Nº outliers: {len(outliers)}")

# 3.3 Aplicar imputación/eliminación

# Ejemplo: imputar rating con mediana
median_rating = df['Review Rating'].median()
df['Review Rating'] = df['Review Rating'].fillna(median_rating)

# Ejemplo: recortar montos al rango [lower, upper]
df['Purchase Amount (USD)'] = np.clip(df['Purchase Amount (USD)'], lower, upper)

# --- 4. Normalización de nombres y valores categóricos --- 

# Renombrar columnas a snake_case
df.rename(columns={
    'Customer Reference ID': 'customer_id',
    'Item Purchased':        'item',
    'Purchase Amount (USD)':  'amount_usd',
    'Date Purchase':         'date_purchase',
    'Review Rating':         'rating',
    'Payment Method':        'payment_method'
}, inplace=True)

# Unificar mayúsculas/minúsculas en categorías
df['item'] = df['item'].str.lower().str.strip()
df['payment_method'] = df['payment_method'].str.lower().str.strip()

# Verificar
print(df.columns)
print(df['payment_method'].unique())

# --- 5 Guardar los datos limpios --- 
df.to_csv('clean_fashion_retail_sales.csv', index=False)

# ================================================================
# 🟢 FASE 2. EXPLORATORY DATA ANALYSIS
# ================================================================

# --- 2.1 Estadísticas descriptivas univariantes --- 
import pandas as pd
import matplotlib.pyplot as plt

# Carga de datos limpios
df = pd.read_csv('clean_fashion_retail_sales.csv', parse_dates=['date_purchase'])

# Verificación de que la columna 'date_purchase' está correctamente convertida a datetime
print(df['date_purchase'].dtype)

# Estadísticas descriptivas
desc = df[['amount_usd', 'rating']].describe()
print(desc)

# Histogramas
plt.figure()
df['amount_usd'].hist(bins=30)
plt.title('Distribución de monto de compra (USD)')
plt.xlabel('Monto (USD)')
plt.ylabel('Frecuencia')
plt.show()

plt.figure()
df['rating'].hist(bins=20)
plt.title('Distribución de review rating')
plt.xlabel('Rating')
plt.ylabel('Frecuencia')
plt.show()

# --- 2.2 Análisis de variables categóricas --- 
# Frecuencia de ítems
item_counts = df['item'].value_counts()
print(item_counts.head(10))

# Gráfico de barras
plt.figure(figsize=(8,4))
item_counts.head(10).plot(kind='bar')
plt.title('Top 10 items comprados')
plt.xlabel('Item')
plt.ylabel('Número de compras')
plt.xticks(rotation=45)
plt.show()

# Frecuencia de método de pago
pay_counts = df['payment_method'].value_counts()
print(pay_counts)

plt.figure()
pay_counts.plot(kind='bar')
plt.title('Métodos de pago')
plt.xlabel('Método')
plt.ylabel('Frecuencia')
plt.show()

# --- 2.3 Tendencias temporales --- 
# Asegúrate de que la columna 'date_purchase' es datetime antes de establecer el índice
df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')

# Establecer la columna 'date_purchase' como índice
df.set_index('date_purchase', inplace=True)

# Verifica que el índice es un DatetimeIndex
print(df.index)

# Resampling con 'ME' (Monthly End) para evitar la deprecación
monthly_sales = df['amount_usd'].resample('ME').sum()

# Ver los resultados del resampling
print(monthly_sales)

# Serie de tiempo
plt.figure(figsize=(10,4))
monthly_sales.plot()
plt.title('Ventas mensuales totales (USD)')
plt.xlabel('Fecha')
plt.ylabel('Ventas USD')
plt.show()

# Ventas por día de la semana
df['weekday'] = df.index.day_name()
weekday_sales = df.groupby('weekday')['amount_usd'].sum().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
plt.figure()
weekday_sales.plot(kind='bar')
plt.title('Ventas por día de la semana')
plt.xlabel('Día')
plt.ylabel('Ventas USD')
plt.show()

# --- 2.4 Correlaciones y relaciones entre variables --- 
import seaborn as sns

# Matriz de correlación
corr = df[['amount_usd', 'rating']].corr()
print(corr)

# Heatmap
plt.figure(figsize=(4,3))
sns.heatmap(corr, annot=True)
plt.title('Correlación entre monto y rating')
plt.show()

# Scatter plot Amount vs. Rating
plt.figure()
plt.scatter(df['rating'], df['amount_usd'], alpha=0.5)
plt.title('Rating vs. monto de compra')
plt.xlabel('Rating')
plt.ylabel('Monto USD')
plt.show()

# --- 2.5 Análisis de clientes activos vs. ocasionales --- 
# Número de compras por cliente
purchase_counts = df.groupby('customer_id').size().rename('purchase_count')
df = df.merge(purchase_counts, on='customer_id')

# Distribución de purchase_count
plt.figure()
df['purchase_count'].hist(bins=20)
plt.title('Distribución de número de compras por cliente')
plt.xlabel('Compras')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot de amount_usd por purchase_count grouping
plt.figure(figsize=(6,4))
sns.boxplot(x=pd.cut(df['purchase_count'], bins=[0,1,3,10,100], labels=['1','2–3','4–10','>10']),
            y=df['amount_usd'])
plt.title('Monto de compra vs. número de compras')
plt.xlabel('Compras por cliente')
plt.ylabel('Monto USD')
plt.show()

# ================================================================
# 🟢 FASE 3. RFM ANALYSIS
# ================================================================

# --- 3.1 Definir fecha de referencia (“snapshot_date”) --- 
import pandas as pd

# Lee el CSV
df = pd.read_csv('clean_fashion_retail_sales.csv')

# Convierte usando el formato día-mes-año
df['date_purchase'] = pd.to_datetime(
    df['date_purchase'],
    format='%d-%m-%Y'
)

# Verificación
print(df['date_purchase'].dtype)

snapshot_date = df['date_purchase'].max() + pd.Timedelta(days=1)
print("Snapshot date:", snapshot_date.date())

# --- 3.2 Calcular Recencia, Frecuencia y Valor monetario --- 

# Agrupar por cliente
rfm = df.groupby('customer_id').agg({
    'date_purchase': lambda x: (snapshot_date - x.max()).days,
    'customer_id': 'count',
    'amount_usd': 'sum'
})

# Renombrar columnas
rfm.rename(columns={
    'date_purchase': 'recency',
    'customer_id': 'frequency',
    'amount_usd': 'monetary'
}, inplace=True)

# Ver ejemplo
print(rfm.head())

# --- 3.3 Asignar scores (1–5) usando quintiles --- 
# Función para asignar quintiles
def rfm_score(x, quantiles, col, reverse=False):
    if reverse:
        # para recencia: menor día = mejor score
        return pd.qcut(x[col], 5, labels=[5,4,3,2,1])
    else:
        # para frecuencia y monetario: mayor valor = mejor score
        return pd.qcut(x[col], 5, labels=[1,2,3,4,5])

# Calcular cuantiles
quantiles = rfm.quantile([0.2,0.4,0.6,0.8]).to_dict()

# Asignar R_score, F_score, M_score
rfm['R_score'] = rfm.apply(lambda row: rfm_score(rfm, quantiles, 'recency', reverse=True).loc[row.name], axis=1)
rfm['F_score'] = rfm.apply(lambda row: rfm_score(rfm, quantiles, 'frequency').loc[row.name], axis=1)
rfm['M_score'] = rfm.apply(lambda row: rfm_score(rfm, quantiles, 'monetary').loc[row.name], axis=1)

# Convertir scores a enteros
rfm[['R_score','F_score','M_score']] = rfm[['R_score','F_score','M_score']].astype(int)

print(rfm.head())

# --- 3.4 Construir el código de segmento RFM --- 
# Concatenar scores en string
rfm['RFM_Score'] = (rfm['R_score'].astype(str) + 
                    rfm['F_score'].astype(str) + 
                    rfm['M_score'].astype(str))
rfm['RFM_Score'].head()

# --- 3.5 Clasificar segmentos con etiquetas legibles --- 
# Definir función de etiquetado simple
def label_segment(row):
    if row['RFM_Score'] == '555':
        return 'Top Champions'
    if int(row['R_score']) >= 4 and int(row['F_score']) >= 4:
        return 'Champions'
    if int(row['R_score']) <= 2 and int(row['F_score']) <= 2:
        return 'At Risk'
    if int(row['R_score']) >= 4:
        return 'Recent Buyers'
    if int(row['F_score']) >= 4:
        return 'Frequent Buyers'
    return 'Others'

rfm['segment'] = rfm.apply(label_segment, axis=1)
rfm['segment'].value_counts()

# --- 3.6 Análisis de perfiles RFM --- 
# Estadísticas por segmento
profile = rfm.groupby('segment').agg({
    'recency':   ['mean','median'],
    'frequency': ['mean','median'],
    'monetary':  ['mean','median','count']
}).round(1)
profile

# Mostrar conteos de RFM_Score y segmentos
print("Distribución de RFM_Score:")
print(rfm['RFM_Score'].value_counts(), "\n")

print("Distribución de segmentos:")
print(rfm['segment'].value_counts(), "\n")

# Imprimir el perfil agregado por segmento
print("Perfil RFM por segmento:")
print(profile, "\n")

# Gráficos para visualizar 
import matplotlib.pyplot as plt

# Bar chart segmentos
plt.figure(figsize=(6,4))
rfm['segment'].value_counts().plot(kind='bar')
plt.title('Clientes por segmento')
plt.xlabel('Segmento')
plt.ylabel('Número de clientes')
plt.tight_layout()
plt.show()

# Histograma de RFM_Score
plt.figure(figsize=(6,4))
rfm['RFM_Score'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribución de RFM_Score')
plt.xlabel('RFM Score')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# ================================================================
# 🟢 FASE 4 CLUSTERING DE CLIENTES
# ================================================================

# --- 4.1 Selección y preparación de features --- 
# Partimos del DataFrame rfm de la Fase 3
features = rfm[['R_score', 'F_score', 'M_score']].copy()

# Añadir rating promedio por cliente
rating_mean = df.groupby('customer_id')['rating'].mean().rename('rating_mean')
features = features.merge(rating_mean, left_index=True, right_index=True)

# Vista previa
print(features.head())


# --- 4.2 Estandarizar las variables --- 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Convertir de nuevo a DataFrame para inspección
import pandas as pd
X_scaled = pd.DataFrame(X_scaled, index=features.index, columns=features.columns)
print(X_scaled.describe().round(2))


# --- 4.3 Determinar el número óptimo de clusters --- 
# 4.3.1 Método del codo (Elbow)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(K, wcss, 'o-')
plt.title('Elbow Method')
plt.xlabel('Número de clusters k')
plt.ylabel('WCSS (Inertia)')
plt.show()

# 4.3.2 Silhouette Score
from sklearn.metrics import silhouette_score

sil = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil.append(silhouette_score(X_scaled, labels))

plt.figure()
plt.plot(K, sil, 'o-')
plt.title('Silhouette Score vs k')
plt.xlabel('Número de clusters k')
plt.ylabel('Silhouette Score')
plt.show()

# --- 4.4 Entrenar K-Means y asignar etiquetas --- 
# Elegimos k = 4
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42)
rfm['cluster'] = kmeans.fit_predict(X_scaled)

# Ver distribución de clientes por cluster
print(rfm['cluster'].value_counts())

# --- 4.5 Caracterizar clusters --- 
# Centroides en escala original: inversa de la transformación
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids = pd.DataFrame(centroids, columns=features.columns)
print("Centroides (R_score, F_score, M_score, rating_mean si aplica):")
print(centroids.round(2))

# Estadísticas agregadas por cluster
cluster_profile = rfm.groupby('cluster').agg({
    'recency':   ['mean','median'],
    'frequency': ['mean','median'],
    'monetary':  ['mean','median'],
    'R_score':   'mean',
    'F_score':   'mean',
    'M_score':   'mean',
    'segment':   'count'
}).round(1)
print(cluster_profile)

# --- 4.6 Visualización de clusters --- 
import matplotlib.pyplot as plt

# 2D: R_score vs F_score
plt.figure(figsize=(6,4))
for c in range(k_opt):
    subset = X_scaled[rfm['cluster']==c]
    plt.scatter(subset['R_score'], subset['F_score'], label=f'Cluster {c}', alpha=0.6)
plt.xlabel('R_score (estandarizado)')
plt.ylabel('F_score (estandarizado)')
plt.legend()
plt.title('Clusters en R vs F')
plt.show()

# Boxplot de monetary por cluster
plt.figure(figsize=(6,4))
import seaborn as sns
sns.boxplot(x=rfm['cluster'], y=rfm['monetary'])
plt.title('Monetary por cluster')
plt.xlabel('Cluster')
plt.ylabel('Gasto total USD')
plt.show()

# ================================================================
# 🟢 FASE 5: VISUALIZACIÓN DE SEGMENTOS
# ================================================================

# --- 5.1 Scatter plot R_score vs F_score coloreado por cluster --- 
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
for c in sorted(rfm['cluster'].unique()):
    subset = rfm[rfm['cluster']==c]
    plt.scatter(subset['R_score'], subset['F_score'],
                label=f'Cluster {c}', alpha=0.6)
plt.xlabel('R_score')
plt.ylabel('F_score')
plt.title('Clusters: R_score vs F_score')
plt.legend()
plt.tight_layout()
plt.show()


# --- 5.2 Boxplot de Monetary (gasto) por cluster --- 
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(x='cluster', y='monetary', data=rfm)
plt.title('Distribución de gasto total (monetary) por cluster')
plt.xlabel('Cluster')
plt.ylabel('Gasto Total (USD)')
plt.tight_layout()
plt.show()


# --- 5.3 Tamaño de cada cluster (bar chart) --- 
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
rfm['cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('Número de clientes por cluster')
plt.xlabel('Cluster')
plt.ylabel('Cantidad de clientes')
plt.tight_layout()
plt.show()

