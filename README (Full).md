# <h1 align="center">Segmentación de clientes para Fashion Retail Co. </h1>

# 1. Resumen ejecutivo

> Problema de negocio:
> 
> 
> Las promociones de Fashion Retail Co. son genéricas, su ROI es bajo y la tasa de desuscripción de campañas es alta, lo que refleja una fidelidad insuficiente de los clientes.
> 
> **Solución propuesta:**
> 
> Implementamos un análisis RFM (Recencia, Frecuencia, Monetario) y clustering K-Means (k=4) mediante el script `FRS.py` para identificar segmentos clave: “Champions” de alto valor, “At Risk” para re-enganche, y grupos intermedios para acciones de cross-sell y up-sell 
> 
> **Hallazgos clave:**
> 
> - “Champions” (cluster 1) presentan un gasto mediano de $2 200 USD y un ROI simulado de +143 %.
> - “At Risk” (cluster 2) responde a cupones con un ROI de +667 %.
> - Dos clusters concentran ~100 clientes, donde pequeñas inversiones pueden generar altos retornos
> 
> **Recomendaciones:**
> 
> 1. Automatizar segmentación RFM mensual para alimentar campañas dinámicas.  
> 2. Desplegar bundles en el rango \$40–60 USD y un programa de fidelidad con referidos.
> 3. Diseñar acciones específicas por cluster (VIP, re-enganche, A/B tests) y medir continuamente KPI de retención y ROI.

---

# 2. Contexto y objetivos

## 2.1 Contexto del proyecto

- **Empresa:** Fashion Retail Co.
- **Sector:** Retail de moda (prendas y accesorios).
- **Canales de venta:**
    - Tiendas físicas (punto de venta).
    - E-commerce (sitio web y app móvil).
- **Datos y código:**
    - Raw data: `Fashion_Retail_Sales.csv`.
    - Clean data: `clean_fashion_retail_sales.csv`
    - Script principal: `FRS.py` (preprocesamiento, RFM, clustering)

## 2.2 Problema de negocio y oportunidad

| Problema actual | Oportunidad |
| --- | --- |
| • Promociones y comunicaciones genéricas → ROI bajo. | • Segmentar “Champions” (alto valor) para campañas de lifetime-value. |
| • Alta tasa de desuscripción y poca fidelidad. | • Re-enganchar “At Risk” con ofertas personalizadas y cupones agresivos. |
| • Falta de método sistemático para distinguir clientes por valor. | • Optimizar cross-sell y up-sell mediante recomendaciones adaptadas al perfil RFM y cluster. |

## 2.3 Objetivos del proyecto

| Tipo | Objetivo |
| --- | --- |
| **General** | Segmentar la base de clientes para optimizar marketing, retención y ventas. |
| **Específicos** | 1. Fase 1: Limpieza y EDA para entender calidad y patrones de los datos.
2. Fase 2: EDA temporal y categórico. 
3. Fase 3: Cálculo RFM. 
4. Fase 4: Clustering K-Means (k=4) y validación (Elbow, Silhouette).
5. Fase 5: Visualización de segmentos (scatter, boxplots, distribución).
6. Fase 6: Recomendaciones por segmento y simulación de ROI. |

# 3. Limpieza y preparación (Fase 1)

Antes de realizar cualquier movimiento o alteración a la información, es importante subir y limpiar ésta.

## 3.1 Importación de librerías y carga del CSV

Al inicio se preparan las herramientas que vamos a usar:

1. **`import os` y `import pandas as pd`**
    - **`os`** nos permite manejar rutas de archivos de forma independiente del sistema operativo.
    - **`pandas`** (alias `pd`) es la biblioteca principal para manipular datos tabulares en Python.
2. **Determinación de la ruta del archivo**
    
    ```python
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(base_path, 'Fashion_Retail_Sales.csv')
    ```
    
    - Se halla la carpeta donde está el script (`base_path`).
    - A partir de ahí, se construye la ruta completa al CSV (`csv_path`), asegurando que funcione igual en Windows, macOS o Linux.
3. **Lectura del CSV en un DataFrame**
    
    ```python
    df = pd.read_csv(csv_path)
    ```
    
    - `df` es la tabla de datos en memoria: cada fila es una venta y cada columna una variable (cliente, monto, fecha, etc.).
4. **Exploración inicial**
    
    ```python
    display(df.head()); display(df.tail())
    print(df.shape); print(df.dtypes)
    ```
    
    - `head()` muestra las primeras 5 filas; `tail()` las últimas 5.
    - Con `shape` vemos cuántas filas y columnas hay.
    - `dtypes` indica el tipo de dato de cada columna (p.ej. numérico, texto).
        
        Esto nos da una “vista rápida” de la calidad y estructura de los datos antes de tocar nada.
        

---

## 3.2. Conversión de fechas y creación de variables temporales

Las fechas suelen venir como texto; para analizarlas es mejor convertirlas a un tipo fecha:

1. **Parseo de la columna de fecha**
    
    ```python
    df['date_purchase'] = pd.to_datetime(df['Date Purchase'], format='%d-%m-%Y')
    ```
    
    - Toma la cadena “DD-MM-YYYY” y la convierte a un objeto `datetime64`, que entiende días, meses, años.
2. **Verificación**
    
    ```python
    print(df['date_purchase'].dtype)
    ```
    
    - Se comprueba que ahora el tipo es `datetime64[ns]`.
3. **Extracción de componentes**
    
    ```python
    df['year']    = df['date_purchase'].dt.year
    df['month']   = df['date_purchase'].dt.month
    df['day']     = df['date_purchase'].dt.day
    df['weekday'] = df['date_purchase'].dt.day_name()
    ```
    
    - Se crean columnas auxiliares (`year`, `month`, `day`, `weekday`) que luego facilitan análisis por periodos (ventas mensuales, ventas por día de la semana, etc.).

---

## 3.3 Detección y tratamiento de valores faltantes y outliers

### 3.3.1 Valores nulos

```python
print(df.isna().sum())
```

- Cuenta cuántos valores “vacíos” hay en cada columna.
- Sirve para decidir si se llenan o eliminan esas filas.

### 3.3.2 Outliers en el monto de la compra

1. **Resumen estadístico**
    
    ```python
    print(df['Purchase Amount (USD)'].describe())
    ```
    
    - Muestra media, percentiles, mínimo y máximo; ayuda a ver si hay valores extremadamente altos o bajos.
2. **Regla IQR**
    
    ```python
    Q1 = df['Purchase Amount (USD)'].quantile(0.25)
    Q3 = df['Purchase Amount (USD)'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df['Purchase Amount (USD)']<lower) | (df['Purchase Amount (USD)']>upper)]
    ```
    
    - El **IQR** (rango intercuartílico) marca el rango “normal”.
    - Valores fuera de `[Q1−1.5·IQR, Q3+1.5·IQR]` se consideran atípicos.
3. **Imputación y recorte (clipping)**
    
    ```python
    median_rating = df['Review Rating'].median()
    df['Review Rating']    = df['Review Rating'].fillna(median_rating)
    df['Purchase Amount (USD)'] = np.clip(df['Purchase Amount (USD)'], lower, upper)
    ```
    
    - Las valoraciones (`Review Rating`) vacías se llenan con la mediana (valor típico).
    - Los montos fuera de rango se “recortan” al límite inferior o superior, evitando que esos outliers distorsionen análisis posteriores.

---

## 3.4 Normalización de nombres y valores categóricos

Para escribir un código limpio y consistente:

1. **Se renombran columnas a `snake_case`**
    
    ```python
    df.rename(columns={
      'Customer Reference ID': 'customer_id',
      'Item Purchased':        'item',
      'Purchase Amount (USD)':  'amount_usd',
      'Date Purchase':         'date_purchase',
      'Review Rating':         'rating',
      'Payment Method':        'payment_method'
    }, inplace=True)
    ```
    
    - Facilita referirse a columnas en Python (sin espacios ni caracteres especiales).
2. **Estandarizar texto en categorías**
    
    ```python
    df['item']           = df['item'].str.lower().str.strip()
    df['payment_method'] = df['payment_method'].str.lower().str.strip()
    ```
    
    - Se convierte todo a minúsculas y elimina espacios extra, unificando variantes (“Credit Card ” vs. “credit card”).

---

## 3.5 Guardado del dataset limpio

```python
df.to_csv('clean_fashion_retail_sales.csv', index=False)
```

- Se exporta el DataFrame ya procesado a un nuevo CSV.
- Este archivo será la base para las fases siguientes (RFM, clustering, visualizaciones)

# 4. Análisis exploratorio de datos (EDA) (Fase 2)

## 4.1 Estadísticas descriptivas univariantes

1. **Carga del dataset limpio**
    
    ```python
    df = pd.read_csv('clean_fashion_retail_sales.csv', parse_dates=['date_purchase']
    ```
    
    - Se vuelve a leer el CSV que guardamos en la Fase 1.
    - Con `parse_dates` aseguramos que `date_purchase` ya entre directamente como fecha.
2. **Verificación de tipo**
    
    ```python
    print(df['date_purchase'].dtype)
    ```
    
    - Se comprueba que esa columna sea `datetime64[ns]`, requisito para poder hacer series de tiempo y extraer componentes temporales sin errores.
3. **Resumen numérico**
    
    ```python
    desc = df[['amount_usd','rating']].describe()
    print(desc)
    ```
    
    - `.describe()` calcula para cada variable:
        - **count** (número de registros no nulos)
        - **mean** (promedio)
        - **std** (desviación estándar)
        - **min**, **25%**, **50%** (mediana), **75%**, **max**
    - Con esto vemos, por ejemplo, si la distribución de montos o de ratings está muy sesgada o si hay valores extremos.
4. **Histogramas**
    
    ```python
    df['amount_usd'].hist(bins=30)
    df['rating'].hist(bins=20)
    ```
    
    - Permiten visualizar la forma de la distribución:
        - ¿Hay muchos clientes haciendo compras muy pequeñas?
        - ¿La mayoría da rating “5” o hay variedad?
    - Cada “bin” agrupa un rango de valores y nos muestra cuántos caen dentro.

## Analizando la distribución del monto de compra y el review rating

<div align="center">
  <img src="https://github.com/user-attachments/assets/d39d2b57-0b65-4f8f-b9d1-28fdc8efad0b" alt="1. Distribución de compra (USD)s" width="1280" height="720" />
  <div align="center">
    Distribución de compra (USD)
  </div>
</div>

### Gráfico 1. “Distribución de monto de compra (USD)”

**Datos duros**

- La gran mayoría de transacciones se concentran entre **$10 y $200 USD**.
- Hay un pico notable alrededor de $40–$60 USD y otro más disperso entre $120–$180 USD.
- Aparece un pequeño “isla” de compras alrededor de **$300 USD** (unos 40–45 transacciones), claramente por encima del rango general.

**Histiograma**

- **Compradores cotidianos**: La mayoría de pedidos son de valor moderado ($40–$60), tal vez camisetas, accesorios o complementos de precio medio.
- **Compras más elevadas** ($120–$180): Probablemente corresponden a conjuntos completos o prendas de mayor ticket (ropa de temporada, chaquetas, conjuntos).
- **Mega-compras (~$300)**: Un puñado de clientes hace pedidos “big-basket” —tal vez varias piezas de alto valor o compras corporativas— que tiran del histograma hacia la derecha.

**Implicaciones del negocio**

- **A las compras del segmento masivo** (picos centrales), se pueden emplear campañas de productos populares y bundles en $40–$60.
- **A las compras del segmento premium** ($120+), se le pueden emplear ofertas exclusivas o se puede hacer upsell de artículos de mayor margen.
- **A las compras de alto gasto** (~$300), los clientes VIP; se les puede incluir en un programa de fidelidad especial o atención personalizada.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5f91a98e-50ab-4f46-b9d2-bc9aa4e29efa" alt="Distribución del review rating" width="1280" height="720" />
  <div align="center">
    Distribución del review rating
  </div>
</div>

### Gráfico 2. “Distribución del review rating”

**Datos duros**

- El rating más frecuente es exactamente **3.0**, con un gran “pico” (más de 450 valoraciones).
- Se observan “mesetas” secundarias en torno a **2.5–2.8** y **4.5–5.0**.
- Hay menos valoraciones extremas en 1.0 o 5.0 de lo que esperaríamos si la satisfacción fuera totalmente polarizada.

**Histiograma**

**Cluster central en 3**:

- Muchos clientes usan “3 estrellas” como un “punto medio” neutro: ni muy satisfechos, ni muy insatisfechos

**Sub-picos en 2.5–3 y 4.5–5**:

- Algunos clientes se sienten levemente insatisfechos (2–3), quizás por tallas o expectativas de producto.
- Otro grupo valora muy alto (4.5–5), probablemente quienes quedan encantados con la calidad o el servicio.

**Implicaciones del negocio**

**Zona de mejora** (2–3 estrellas): Analizar qué productos o procesos (envío, devoluciones) generan esa neutralidad/ligera insatisfacción.

- **Acción**: Diseñar encuestas cortas a quienes ponen 3 estrellas para entender qué pequeño ajuste los empujaría a 4–5.
- **Promotores** (4–5 estrellas): Incentivar reviews positivas adicionales y referrals.

---

## 4.2 Análisis de variables categóricas

1. **Frecuencia de productos**
    
    ```python
    item_counts = df['item'].value_counts()
    print(item_counts.head(10))
    ```
    
    - `value_counts()` ordena los ítems por número de compras, de mayor a menor.
    - Vemos el “top 10” de los productos más vendidos.
2. **Gráfico de barras de los 10 principales**
    
    ```python
    item_counts.head(10).plot(kind='bar')
    ```
    
    - Muestra visualmente cuáles son esos 10 ítems más populares, facilitando comparaciones rápidas.
3. **Frecuencia de métodos de pago**
    
    ```python
    pay_counts = df['payment_method'].value_counts()
    pay_counts.plot(kind='bar')
    ```
    
    - Igual que con los ítems, vemos qué formas de pago (efectivo, tarjeta, etc.) son más usadas.

## Analizando los items más comprados y los métodos de pago

<div align="center">
  <img src="https://github.com/user-attachments/assets/37b88b15-f90b-4380-b77e-0e6b958f7837" alt="Top 10 items comprados" width="1280" height="720" />
  <div align="center">
    Top 10 items comprados
  </div>
</div>


### Gráfico 3. “Top 10 items comprados”

**Datos duros**

- El artículo más vendido es **el cinturón** con cerca de 90 compras, seguido de las faldas y los **shorts** en torno a 85–88.
- El top 10 cierra con artículos como **“pijamas”**, **“camisolas”**, **“mocasines”** y **“sudaderas”** rondando las 75–82 ventas.

**Histiograma**

- El cinturón encabeza la lista. Como artículo de complemento (“add-on”), parece ser un favorito fácil de incorporar en el carrito.
- Las faldas y shorts dominan, sugiriendo que buena parte de las ventas proviene de líneas de primavera/verano.
- La presencia de pijamas y sudaderas  en el top 10 indica que el cliente valora tanto moda exterior como prendas cómodas para el hogar.

**Implicaciones del negocio**

- Se recomendaría crear bundles estratégicos como la combinación de cinturones con faldas o pantalones. Se pueden desarrollar ofertas como “compre 1 y lleve descuento en el cinturón” para elevar el ticket promedio.
- Se puede contemplar aumentar el inventario faldas y shorts antes de la época primavera/verano o se pueden planificar campañas “back to loungewear” para las pijamas y sudaderas en otoño/invierno

<div align="center">
  <img src="https://github.com/user-attachments/assets/4796aa4c-bb74-4156-9282-a0b9fe4f21b8" alt="Métodos de pago" width="1280" height="720" />
  <div align="center">
    Métodos de pago
  </div>
</div>

### Gráfico 4. “Métodos de pago”

**Datos duros**

- Hay aproximadamente **1 750 compras con tarjeta de crédito** frente a **1 620 en efectivo**.
- El uso de **tarjetas de cr+edito** supera ligeramente al del efectivo, pero ambos métodos están bien representados.

**Sobre los métodos**

- El hecho de que más de la mitad de las transacciones sean con tarjeta sugiere comodidad y confianza en pagos electrónicos, especialmente en e-commerce.
- Un 48 % aún paga en efectivo (probablemente en tienda física), lo que indica que no podemos descuidar ese canal.

**Implicaciones del negocio**

- Se puede ofrecer un pequeño descuento o “cashback” al usar tarjeta de crédito para fomentar aún más la adopción digital y acelerar el proceso de pago online.
- En tiendas físicas, se pueden optimizar puntos de venta para que el pago con tarjeta sea rápido y sencillo, sin largas esperas que desincentiven su uso.
- Se puede informar a clientes sobre seguridad y beneficios del pago electrónico (programas de puntos, conveniencia), sin eliminar opciones de efectivo para quienes lo prefieren.

---

## 4.3 Tendencias temporales

1. **Asegurar índice de fecha**
    
    ```python
    df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')
    df.set_index('date_purchase', inplace=True)
    ```
    
    - Convertimos (otra vez, por precaución) a datetime y lo establecemos como índice, requisito para usar `resample()`.
2. **Ventas mensuales**
    
    ```python
    monthly_sales = df['amount_usd'].resample('ME').sum()
    ```
    
    - `resample('ME')` agrupa por “fin de mes” y suma los montos.
    - Así obtenemos una serie de tiempo de ventas totales por mes.
3. **Gráfico de serie temporal**
    
    ```python
    monthly_sales.plot(figsize=(10,4))
    ```
    
    - Nos permite ver tendencias:
        - ¿Hay estacionalidad?
        - ¿Crecen las ventas con el tiempo?
        - ¿Detectamos picos (por ejemplo, temporada de vacaciones)?
4. **Ventas por día de la semana**
    
    ```python
    df['weekday'] = df.index.day_name()
    weekday_sales = df.groupby('weekday')['amount_usd'].sum().reindex([...])
    weekday_sales.plot(kind='bar')
    ```
    
    - Agrupamos por nombre de día (“Monday”, “Tuesday”…).
    - Reindexamos en orden lógico de lunes a domingo.
    - Identificamos qué días la tienda vende más.

## Analizando las ventas mensuales y semanales

<div align="center">
  <img src="https://github.com/user-attachments/assets/5529c2c7-7142-4156-87cb-b49b87e6acac" alt="Ventas mensuales totales USD" width="1280" height="720" />
  <div align="center">
    Ventas mensuales totales USD
  </div>
</div>

### Gráfico 5. “Ventas mensuales totales USD”

**Datos duros**

- Este gráfico es una serie de tiempo que muestra, para cada mes, la suma de todos los montos de compra.
- Se aprecian picos y valles: por ejemplo, meses con ventas superiores a $25 000 USD y otros por debajo de $15 000 USD.
- Puede haber una clara estacionalidad: meses de alta actividad (tal vez coincidiendo con rebajas, lanzamientos de colección o festividades) y meses de menor movimiento.

**Sobre la serie temporal**

- Si los picos caen en noviembre–diciembre, probablemente reflejan alguna causa como la temporada navideña y el Black Friday.
- Un valle a mitad de año puede señalar un período tranquilo entre colecciones o antes de la vuelta a clases.
- Un incremento sostenido mes a mes indicaría crecimiento orgánico de la marca.

**Implicaciones del negocio**

- **Planificación de campañas:**
    - Concentrar promociones intensivas justo antes del pico natural (por ejemplo, lanzar una pre-venta en octubre para maximizar noviembre).
    - Introducir ofertas de “mid-season” en los meses más bajos para suavizar la curva.
- **Gestión de inventario:**
    - Aumentar stock de productos populares antes de meses pico para evitar quiebres.
    - Reducir inventario o liquidar prendas en meses valle.
- **Evaluación de tendencias:**
    - Si hay una tendencia al alza general, invertir más en marketing continuo; si es plana, explorar nuevas líneas de producto o canales de venta.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5bd54921-6226-4621-a67f-49b2c0d9c46c" alt="Ventas por día de la semana" width="1280" height="720" />
  <div align="center">
    Ventas por día de la semana
  </div>
</div>

### Gráfico 6. “Ventas por día de la semana”

**Datos duros**

- Las barras representan el total de ventas (USD) agregadas por cada día: de lunes a domingo

**Sobre la semana de ventas**

- **Lunes y domingos** son los días con **mayor volumen de ventas**
    - Esto se puede deber a que tras el fin de semana, la gente busca las ofertas que puede haber para comprar el lunes, o bien, esperan a evaluar presupuesto después del fin de semana.
    - En cuanto a las ventas del domingo, se puede deber a que la gente cuenta con más tiempo para ir a las tiendas físicas o para revisar la ropa en línea. Además, las promociones enviadas el fin de semana pueden generar picos dominicales.
- **Martes y sábados** muestran las **ventas más bajas** de la semana.
    - Esto quizá se pueda a que los martes es un día laboral y las personas no les gusta salir, no les da tiempo o no les llama la atención comprar entre semana.
    - La baja en sábados se puede deber a que el público objetivo o general prefiere hacer cosas más de ocio, diversión o culturales.

**Implicaciones del negocio**

Dentro de las acciones que pudieran contemplarse están las siguientes:

| Acción | Días objetivo |
| --- | --- |
| Promociones “Happy Tuesday” | Martes (para reactivar un día flojo) |
| Ofertas “Weekend Flash” | Sábado (incentivar actividad en e-commerce) |
| Planificación de stock online | Reponer antes del domingo |
| Refuerzo de atención en tienda | Domingo y lunes |
| Envío de newsletters y ofertas | Domingo por la tarde, lunes temprano |

---

## 4.4 Correlaciones y relaciones entre variables

1. **Matriz de correlación**
    
    ```python
    corr = df[['amount_usd','rating']].corr()
    print(corr)
    ```
    
    - El coeficiente de Pearson entre monto y rating nos dice si los clientes que gastan más tienden también a dar mejores (o peores) calificaciones.
2. **Heatmap**
    
    ```python
    sns.heatmap(corr, annot=True)
    ```
    
    - Visualiza la matriz con colores y anotaciones, para “ver de un vistazo” si existe alguna relación fuerte (cercana a 1 o −1) o es prácticamente nula (cerca de 0).
3. **Scatter plot (dispersión)**
    
    ```python
    plt.scatter(df['rating'], df['amount_usd'], alpha=0.5)
    ```
    
    - Cada punto es una transacción: eje X = rating, eje Y = monto.
    - Con `alpha` semitransparente vemos la densidad de puntos.
    - Útil para confirmar visualmente lo que dijo la correlación (¿apunta a nube sin forma o a tendencia ascendente/descendente?).

## Analizando la correlación monto-rating

<div align="center">
  <img src="https://github.com/user-attachments/assets/9d8bee36-ae34-46a4-83cc-85112d49676c" alt="Correlación entre el monto y rating" width="1280" height="720" />
  <div align="center">
    Correlación entre el monto y rating
  </div>
</div>

### Gráfico 7. “Correlación entre el monto y rating”

**Datos duros**

- La celda cruzada “amount_usd ↔ rating” marca un valor de 0.043.
- Un coeficiente de Pearson cercano a 0 indica prácticamente nula correlación lineal entre cuánto gastan los clientes y la puntuación que dan.

| Par de variables | Correlación | Interpretación |
| --- | --- | --- |
| amount_usd vs rating | 0.043 | No hay relación lineal significativa. |

**Sobre el resultado**

- Un valor **≈ 0** significa que, en este conjunto de datos, **gastar más no implica una mejor (o peor) valoración**, y viceversa. Es decir, los clientes que pagan tickets altos no necesariamente son los que dejan 5 estrellas, ni los que pagan poco dejan malas notas.

**Implicaciones del negocio**

Con esta información, se podría decir que en:

1. **Estrategias de precio**
    - Ajustar precios (ofertas o upsell) probablemente **no alteraría** de forma directa las valoraciones de producto.
    - Para mejorar el rating, conviene enfocarse en **calidad**, **envío**, **atención al cliente**, más que en el monto.
2. **Segmentación alternativa**
    - Dado que el rating no discrimina el ticket, se puede recurrir a la segmentación RFM (recencia/frecuencia/ valor monetario) para diseñar promociones y tratar el rating como una **métrica de calidad** independiente.
3. **Monitoreo continuo**
    - Aunque hoy la correlación es baja, conviene revisarla tras cambios de catálogo o ajustes de precio para detectar si en el futuro surge alguna tendencia.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f3a3133d-ff21-4172-8bcf-f41b79ebf007" alt="Rating vs. monto de compra" width="1280" height="720" />
  <div align="center">
    Rating vs. monto de compra
  </div>
</div>

### Gráfico 8 “Rating vs. monto de compra”

**Datos duros**

- Cada punto es una transacción: eje X = rating (1–5), eje Y = monto USD.
- La nube aparece **muy dispersa** en vertical para cada rating, confirmando que los montos varían ampliamente sin seguir un patrón según la valoración.
- Hay outliers de monto alto (~$300) en casi todos los niveles de rating, y clientes que dan 5 estrellas gastan desde $10 hasta $300.

**Sobre el gráfico**

- **Se evidencia una ausencia de banda inclinada, es decir,** no se forma una “línea” ascendente o descendente.
- Tanto un rating bajo (1–2) como un rating alto (4–5) pueden acompañarse de montos pequeños o grandes.

**Implicaciones del negocio**

1. **Decoupling precio–satisfacción**
    - Se puede tratar el rating como un **indicador de experiencia** (calidad del producto/servicio), no como métrica de gasto.
    - Se puede diseñar dos líneas de acción en paralelo:
        - **Optimizar ticket** (a través de bundles, upsell basados en RFM).
        - **Elevar rating** (mejoras de UX, calidad de empaque, post-venta).
2. **Análisis de outliers**
    - Se pueden investigar los casos de montos muy altos con rating bajo (¿qué salió mal en esas ventas premium?).
    - Estos clientes VIP insatisfechos merecen atención prioritaria (encuesta personalizada, servicio al cliente dedicado).
3. **Segmentación cruzada**
    - Se puede hacer cruce de segmentos RFM con buckets de rating (por ejemplo: Champions con rating ≤ 3 vs rating ≥ 4) para diseñar campañas específicas:
        - “Champions insatisfechos” → encuesta de satisfacción + oferta de compensación.
        - “Champions satisfechos” → programa de referidos.

---

## 4.5 Análisis de clientes activos vs. ocasionales

1. **Conteo de compras por cliente**
    
    ```python
    purchase_counts = df.groupby('customer_id').size().rename('purchase_count')
    df = df.merge(purchase_counts, on='customer_id')
    ```
    
    - Para cada `customer_id` contamos cuántas filas (compras) tiene y lo unimos al DataFrame original.
    - Ahora cada registro sabe cuántas veces compró ese cliente en total.
2. **Distribución de número de compras**
    
    ```python
    df['purchase_count'].hist(bins=20)
    ```
    
    - Muestra cuántos clientes hicieron 1 compra, 2, 3…
    - Se puede ver si la mayoría compra sólo una vez o hay una “cola” de clientes frecuentes.
3. **Boxplot de monto vs. frecuencia**
    
    ```python
    sns.boxplot(
      x=pd.cut(df['purchase_count'], bins=[0,1,3,10,100], labels=['1','2–3','4–10','>10']),
      y=df['amount_usd']
    )
    ```
    
    - Agrupa clientes en rangos de frecuencia:
        - “1 compra”, “2–3 compras”, “4–10”, “>10”.
    - Para cada grupo dibuja un diagrama de caja (“boxplot”) de los montos.
    - Así sabemos, por ejemplo, si quienes compran más veces también tienden a gastar más por transacción, o si quizá repiten pero con tickets bajos.

## Analizando los montos y números de compras

<div align="center">
  <img src="https://github.com/user-attachments/assets/c3a6104d-63b3-4dc1-978f-4d8b2f3c1775" alt="Distribución de números de compras por cliente" width="1280" height="720" />
  <div align="center">
    Distribución de números de compras por cliente
  </div>
</div>

### Gráfico 9. “Distribución de números de compras por cliente”

**Datos duros**

- Hay un claro **pico alrededor de 20–24 compras**: la mayoría de los clientes en este dataset compraron entre 20 y 24 veces.
- Muy pocos clientes quedan en los extremos (<12 compras o >30 compras).

**Histiograma**

- **Cliente típico**
    - El “cliente medio” realiza unas **22 compras** en el horizonte de datos.
    - Esto sugiere una base de compradores recurrentes: no son usuarios que compran una sola vez, sino que vuelven casi dos docenas de veces.
- **Cola de compradores**
    - Un pequeño grupo de “heavy shoppers” supera las 30 compras: probablemente clientes corporativos, revendedores o suscriptores muy fieles.
    - En el otro extremo, casi no hay “one-timers”: la limpieza de datos y el periodo analizado filtran a quien sólo compró 1–2 veces.

**Implicaciones del negocio**

| Insight | Acción sugerida |
| --- | --- |
| Alta frecuencia media (20–24) | Implementar un programa de fidelidad para mantener este ritmo de recompra. |
| Grupo “heavy shoppers” (>30) | Identificar a estos clientes VIP para ofertas exclusivas y recompensas especiales. |
| Ausencia de compradores esporádicos | Explorar campañas de adquisición para atraer nuevos clientes, pues la mayoría ya repite compras. |

<div align="center">
  <img src="https://github.com/user-attachments/assets/4474a9f8-3f94-44cc-a7c0-b536da1c0464" alt="Monto de compra vs. número de compras" width="1280" height="720" />
  <div align="center">
    Monto de compra vs. número de compras
  </div>
</div>

### Gráfico 10. “Monto de compra vs. número de compras”

**Datos duros**

- Agrupamos clientes en categorías según su frecuencia:
    - “1” compra (prácticamente inexistente aquí),
    - “2–3” compras,
    - “4–10” compras,
    - “>10” compras.
- Para cada grupo, el boxplot muestra la distribución de **monto por transacción**: mediana, cuartiles y posibles outliers.

**Sobre el boxplot**

- **Clientes 4–10 compras**
    - **Mediana de ticket** mayor (alrededor de $130–$140 USD).
    - Compran montos más consistentes y altos por orden.
- **Clientes >10 compras**
    - **Mediana de ticket** algo más baja (alrededor de $100–$115 USD).
    - Mayor dispersión: algunos gastan poco ($10–$50), otros llegan a $200+.
    - Un outlier cerca de $300 indica que algún cliente frecuente también hace ocasionalmente pedidos muy altos.
- **Grupos 1 y 2–3 compras**
    - Casi no aparecen porque casi todos los clientes compran más de 4 veces en el periodo analizado.

**Implicaciones del negocio**

| Grupo de frecuencia | Comportamiento de ticket | Estrategia |
| --- | --- | --- |
| 4–10 compras | Ticket alto y consistente | Ofertas “bundle premium” para mantener alta la compra promedio. |
| >10 compras | Ticket medio-bajo y variable | Incentivos de upsell (p.ej. recomendaciones personalizadas) para elevar su ticket promedio. |
| Outliers de alto ticket | Ocasionales dentro de frecuentes | Enviar cupones de fidelidad post-compra para repetir mega-pedidos. |

---

# 5. Cálculo RFM (Fase 3)

## 5.1 Definir la “snapshot date” (fecha de referencia)

```python
df = pd.read_csv('clean_fashion_retail_sales.csv')
df['date_purchase'] = pd.to_datetime(df['date_purchase'], format='%d-%m-%Y')
snapshot_date = df['date_purchase'].max() + pd.Timedelta(days=1)
```

1. **Lectura del CSV limpio**
    - Traemos el DataFrame `df` que ya contiene los datos sin ruido (Fase 1).
2. **Convertir la fecha de compra**
    - Aseguramos que la columna `date_purchase` sea un objeto de fecha (`datetime64`), usando el formato día-mes-año.
3. **Elegir el punto de “ahora”**
    - `snapshot_date` se fija como el día siguiente a la última transacción registrada.
    - Esto nos da un “punto cero” constante para medir **recencia** (días desde la última compra).

---

## 5.2 Calcular Recencia, Frecuencia y Valor monetario

```python
rfm = df.groupby('customer_id').agg({
  'date_purchase': lambda x: (snapshot_date - x.max()).days,
  'customer_id':   'count',
  'amount_usd':    'sum'
})
rfm.rename(columns={ ... }, inplace=True)
```

1. **Agrupación por cliente**
    - `df.groupby('customer_id')` agrupa todas las compras de cada cliente.
2. **Cálculo de métricas RFM**
    - **Recencia**:
        - Para cada cliente, restamos la última fecha de compra (`x.max()`) de `snapshot_date` y tomamos el número de días.
        - Un valor pequeño significa que acaba de comprar; un valor grande, que lleva tiempo sin hacerlo.
    - **Frecuencia**:
        - Simplemente el conteo de compras (`count` de `customer_id`).
    - **Valor monetario**:
        - La suma total de lo gastado por ese cliente (`sum` de `amount_usd`).
3. **Renombrado**
    - Cambiamos los nombres de columna a `recency`, `frequency` y `monetary` para claridad.

---

## 5.3 Asignar scores (1–5) usando quintiles

```python
quantiles = rfm.quantile([0.2,0.4,0.6,0.8]).to_dict()

# Para cada métrica, cortamos en 5 grupos iguales:
rfm['R_score'] = rfm_score(..., reverse=True)
rfm['F_score'] = rfm_score(..., reverse=False)
rfm['M_score'] = rfm_score(..., reverse=False)
```

1. **¿Por qué quintiles?**
    - Queremos dividir la distribución de cada métrica en 5 grupos iguales (20 % de clientes en cada uno).
2. **Función `rfm_score`**
    - Se usa `pd.qcut` para asignar etiquetas 1–5 según el rango de cada cliente.
    - Para **recencia**, invertimos la escala (menor días = mejor score 5).
    - Para **frecuencia** y valor **monetario**, mayor valor = mejor score 5.
3. **Aplicación**
    - Calculamos los cuantiles globales y luego, fila a fila, asignamos `R_score`, `F_score` y `M_score`.
    - Convertimos esas etiquetas a enteros para poder concatenarlas y analizarlas fácilmente.

---

## 5.4 Construir el código de segmento RFM

```python
rfm['RFM_Score'] = (
   rfm['R_score'].astype(str) +
   rfm['F_score'].astype(str) +
   rfm['M_score'].astype(str)
)
```

- Concatenamos los tres dígitos en una cadena, por ejemplo `"5" + "3" + "4" = "534"`.
- Ese código resume en una sola columna el perfil RFM de cada cliente.

---

## 5.5 Etiquetar segmentos con nombres legibles

```python
def label_segment(row):
    if row['RFM_Score']=='555':      return 'Top Champions'
    if row['R_score']>=4 and row['F_score']>=4: return 'Champions'
    if row['R_score']<=2 and row['F_score']<=2: return 'At Risk'
    if row['R_score']>=4:             return 'Recent Buyers'
    if row['F_score']>=4:             return 'Frequent Buyers'
    return 'Others'

rfm['segment'] = rfm.apply(label_segment, axis=1)
```

- Definimos reglas de negocio sencillas para agrupar clientes:
    - **Top champions**: los mejores en R, F y M (555).
    - **Champions**: muy recientes y muy frecuentes.
    - **At risk**: ni recientes ni frecuentes.
    - **Recent buyers**: compraron muy recientemente.
    - **Frequent buyers**: compran muchas veces.
    - **Others**: el resto.

---

## 5.6 Análisis de perfiles RFM

```python
profile = rfm.groupby('segment').agg({
   'recency':   ['mean','median'],
   'frequency': ['mean','median'],
   'monetary':  ['mean','median','count']
}).round(1)
```

1. **Agrupación y estadísticas**
    - Para cada segmento calculamos:
        - Media y mediana de **recency** (días).
        - Media y mediana de **frequency** (número de compras).
        - Media, mediana y tamaño de muestra de **monetary** (USD).
2. **Interpretación**
    - Estas cifras describen el “perfil tipo” de cada grupo.
3. **Distribuciones**
    - También imprimimos la cuenta de cada `RFM_Score` y de cada `segment` para ver cuántos clientes hay en cada categoría.

---

## 5.7 Visualización de resultados (opcional)

```python
rfm['segment'].value_counts().plot(kind='bar')
rfm['RFM_Score'].value_counts().sort_index().plot(kind='bar')
```

- **Bar chart de segmentos**: número de clientes en “Champions”, “At Risk”, etc.
- **Histograma de RFM_Score**: cuántos clientes hay en cada combinación “XYZ”.

Estos gráficos confirman visualmente la distribución y ayudan a tomar decisiones (por ejemplo, si tenemos pocos “Top Champions”, quizás valga la pena diseñar una campaña para crear más).

## Analizando los clientes por segmento y el RFM Score

<div align="center">
  <img src="https://github.com/user-attachments/assets/943325c5-5388-460a-9782-10ba6383d20f" alt="Clientes por segmento" width="1280" height="720" />
  <div align="center">
    Clientes por segmento
  </div>
</div>

### **Gráfico 11. “Clientes por segmento”**

**Datos duros**

- Recent buyers: 48 clientes
- Others: 39 clientes
- At risk: 32 clientes
- Champions: 23 clientes
- Frequent buyers: 20 clientes
- Top champions: 4 clientes

**Sobre el gráfico**

- **Recent buyers** domina: muchos clientes han comprado muy recientemente, pero aún no sabemos si repetirán.
- Un gran bloque de **Others** agrupa comportamientos mixtos que no encajan en categorías extremas.
- **At risk** muestra un grupo significativo que lleva tiempo sin comprar y puede perderse.
- **Champions** y **Frequent buyers** son grupos valiosos por su recurrencia y gasto.
- Sólo 4 **Top champions** representan a los clientes de élite con máxima recencia, frecuencia y gasto.

**Implicaciones del negocio**

---

Entre las estrategias que se pudiera contemplar según el segmento, encontramos:

| Segmento | Estrategia |
| --- | --- |
| Recent buyers | Campaña de nurturing: recomendaciones post-compra y upsell ligero. |
| Others | Test A/B de mensajes para activar comportamientos deseados. |
| At risk | Cupón de re-enganche (“Te extrañamos: 15 % off”). |
| Champions | Programa VIP: acceso anticipado, regalos exclusivos. |
| Frequent buyers | Incentivos de fidelidad: puntos extra, descuentos escalonados. |
| Top champions | Atención personalizada: gestor de cuenta, invitaciones especiales. |

<div align="center">
  <img src="https://github.com/user-attachments/assets/e633dca5-d01d-45c6-9c64-88c4537deedf" alt="Distribución de RFM_Score" width="1280" height="720" />
  <div align="center">
    Distribución de RFM_Score
  </div>
</div>

### Gráfico 12. “Distribución de RFM_Score”

**Datos duros**

- Más de 30 códigos distintos
- La mayoría de códigos aparece sólo 1–3 veces.
- Un puñado de códigos llega a 5 o hasta 8 repeticiones.

**Sobre el gráfico**

- Existe una **larga cola** de micro-perfiles RFM: cada cliente tiene una combinación única de recencia, frecuencia y gasto.
- Unos pocos perfiles son relativamente comunes (picos de hasta 8 clientes).
- La dispersión justifica agrupar esos códigos en segmentos mayores para simplificar la acción.

**Implicaciones del negocio**

---

Entre las estrategias que se pudiera contemplar están las siguientes:

| Insight | Acción |
| --- | --- |
| Alta diversidad de códigos RFM | Mantener la segmentación simplificada (Champions, At Risk, etc.). |
| Perfiles RFM frecuentes (picos) | Analizar esos perfiles: qué productos compran y qué promociones recibieron. |
| Códigos únicos | Evaluar campañas personalizadas o usar modelos de ML para recomendaciones one-to-one. |

---

# 6. Clustering K-Means (k=4) y validación (Elbow, Silhouette) (Fase 4)

## 6.1 Selección y preparación de las “features”

```python
features = rfm[['R_score','F_score','M_score']].copy()
rating_mean = df.groupby('customer_id')['rating'].mean().rename('rating_mean')
features = features.merge(rating_mean, left_index=True, right_index=True)
```

- **¿Qué hay en `rfm`?**
    - `rfm` es un DataFrame cuya fila es cada cliente y cuyas columnas incluyen las métricas R_score, F_score, M_score (integers 1–5) y otras variables de perfil.
- **Selección de variables**
    - Tomamos esas tres puntuaciones RFM como base del clustering.
    - Opcionalmente añadimos `rating_mean`, el promedio de la valoración (“rating”) que cada cliente dio, para incorporar una dimensión de satisfacción al agrupamiento.
- **Resultado**
    
    Un nuevo DataFrame `features` donde cada fila es un cliente y las columnas son los valores numéricos que usaremos para agrupar (“features”).
    

---

## 6.2 Estandarizar las variables

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_scaled = pd.DataFrame(X_scaled, index=features.index, columns=features.columns)
```

- **¿Por qué estandarizar?**
    
    R_score, F_score y M_score (y rating_mean) pueden tener escalas o rangos distintos. Si no estandarizamos, una variable de mayor magnitud dominará el cálculo de distancia en el clustering.
    
- **`StandardScaler`**
    
    Transforma cada columna para que tenga media 0 y desviación estándar 1. Así todas las features “pesan” igual.
    
- **Reconstrucción de DataFrame**
    
    Convertimos el array resultante de nuevo a un DataFrame con los mismos índices y nombres de columna, para facilitar inspección y unión posterior.
    

---

## 6.3 Determinar el número óptimo de clusters

### 6.3.1 Método del codo (Elbow)

```python
from sklearn.cluster import KMeans
wcss, K = [], range(1,11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    wcss.append(km.inertia_)
plt.plot(K, wcss, 'o-')
```

- **WCSS (Within-Cluster Sum of Squares)**
    
    Mide la suma de las distancias cuadradas de cada punto a su centroide.
    
- **Elbow method**
    
    Se grafica WCSS vs. número de clusters k. Buscamos el “codo” donde la mejora (caída de WCSS) se amortigua. Ese k es un buen compromiso entre complejidad y cohesión de clusters.
    

### 6.3.2 Silhouette Score

```python
from sklearn.metrics import silhouette_score
sil = []
for k in range(2,11):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
    sil.append(silhouette_score(X_scaled, labels))
plt.plot(range(2,11), sil, 'o-')
```

- **Silhouette Score**
    
    Mide, para cada punto, qué tan similar es a su propio cluster comparado con el más cercano. Va de –1 a +1; valores cercanos a +1 indican clusters bien aislados.
    
- **Selección de k**
    
    Buscamos el k con mayor silueta promedio, lo que indica particiones más definidas.
    

---

## 6.4 Entrenar K-Means y asignar etiquetas

```python
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42)
rfm['cluster'] = kmeans.fit_predict(X_scaled)
```

- **Elección de k = 4**
    
    Basada en la combinación de Elbow y Silhouette.
    
- **`fit_predict`**
    
    Ajusta el modelo K-Means a los datos estandarizados y retorna, para cada cliente, el índice de cluster (0, 1, 2 o 3).
    
- **Almacenamiento**
    
    Creamos una nueva columna `cluster` en el DataFrame `rfm` para poder analizar y visualizar posteriormente.
    

---

## 6.5 Caracterizar los clusters

```python
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_profile = rfm.groupby('cluster').agg({
    'recency':['mean','median'], 'frequency':['mean','median'],
    'monetary':['mean','median'], 'R_score':'mean','F_score':'mean','M_score':'mean','segment':'count'
}).round(1)
```

1. **Centroides en escala original**
    - Aplicamos la inversa de la estandarización para leer los valores de R_score, F_score, M_score y rating_mean de cada centroide en su escala natural.
    - Estos valores “típicos” describen el cliente promedio de cada cluster.
2. **Perfil agregado**
    - Calculamos media y mediana de recency/frequency/monetary para los clientes de cada cluster.
    - También promediamos los scores R_score, F_score, M_score y contamos cuántos clientes hay (`segment: count`).
3. **Interpretación**
    - Cada fila de `cluster_profile` resume el comportamiento de un cluster:

---

## 6.6 Visualización de clusters

### 6.6.1 Scatter 2D: R_score vs F_score

```python
for c in range(k_opt):
    subset = X_scaled[rfm['cluster']==c]
    plt.scatter(subset['R_score'], subset['F_score'], label=f'Cluster {c}')
```

- Cada punto es un cliente, posicionado según sus scores estandarizados de Recencia y Frecuencia.
- **Objetivo**
    
    Ver solapamientos o separaciones entre clusters:
    
    - Un cluster puede agruparse en la esquina “alto R, alta F” (Champions potenciales).
    - Otro puede estar en “bajo R, baja F” (clientes en riesgo).

### 6.6.2 Boxplot de monetary por cluster

```python
sns.boxplot(x=rfm['cluster'], y=rfm['monetary'])
```

- Para cada cluster, la distribución de gasto total (monetary).
- **Objetivo**
    
    Comparar rápidamente qué clusters gastan más:
    
    - Un cluster con mediana alta es tu grupo premium.
    - Clusters con colas bajas podrían necesitar acciones de upsell.

## Analizando el Elbow Method, el Silhouette Score, los clusters en R vs. F y el valor monetario por clusters

<div align="center">
  <img src="https://github.com/user-attachments/assets/ea932915-a2be-41d1-b7c4-5fbc11d88fa5" alt="Elbow Method" width="1280" height="720" />
  <div align="center">
    Elbow Method
  </div>
</div>

### Gráfico 13. “**Elbow Method**”

**Datos duros**

- Eje X: número de clusters *k* (1 a 10).
- Eje Y: WCSS (Within-Cluster Sum of Squares), la inercia total—es decir, cuán “apretados” están los puntos alrededor de sus centroides.
- La curva desciende rápido de *k = 1* a *k = 3*, luego la pendiente se suaviza. A partir de *k ≈ 4*–5 la mejora marginal en WCSS es pequeña.

**Sobre el gráfico**

- **De 1 a 3 clusters**: gran caída de inercia, lo que significa que separar en 2 o 3 grupos aporta mucha ganancia en cohesión interna.
- **A partir de 4 clusters**: el “codo” o punto donde la curva se “dobla” indica que añadir más clusters no reduce sustancialmente la variabilidad dentro de cada cluster.
- Ese punto de inflexión (codo) suele recomendar **k = 4** como compromiso entre simplicidad (menos clusters) y calidad de agrupamiento (baja inercia).

**Implicaciones del negocio**

Contemplando lo anterior, se puede mencionar lo siguiente:

| Insight | Acción |
| --- | --- |
| Gran ganancia al pasar de 1→3 clusters | Ya no basta con un solo grupo: hay al menos tres perfiles de cliente claramente distintos. |
| “Codo” en k=4 | Elegir 4 clusters para capturar variedad de comportamientos sin sobredimensionar el modelo. |
| Poca mejora >k=5 | Evitar más de 4–5 clusters: complejidad innecesaria para marketing y gestión de segmentos. |

<div align="center">
  <img src="https://github.com/user-attachments/assets/f4ee522f-4816-4464-b8db-4b69210b91cd" alt="Silhouette Score vs K" width="1280" height="720" />
  <div align="center">
    Silhouette Score vs K
  </div>
</div>

### Gráfico 14. “**Silhouette Score”**

**Datos duros**

- Eje X: número de clusters *k* (2 a 10).
- Eje Y: Silhouette Score promedio (rango –1 a +1).
- El valor de silueta sube desde *k = 2* hasta un **máximo alrededor de k = 4**, luego tiende a bajar o estabilizarse.

**Sobre el gráfico**

- **k=2–3**: silueta moderada, los clusters empiezan a formarse pero aún se solapan.
- **k=4**: pico de silueta, indica la partición más “limpia”—los clientes dentro de cada cluster son muy similares y bien separados de otros clusters.
- **k>4**: la silueta desciende, lo que sugiere que añadir más grupos empieza a crear divisiones artificiales con menos coherencia interna.

**Implicaciones del negocio**

Contemplando lo anterior, se puede mencionar lo siguiente:

---

| Insight | Acción |
| --- | --- |
| Silhouette máxima en k=4 | Confirma que 4 clusters es la elección óptima para segmentación. |
| Pérdida de cohesión >4 clusters | Evitar subdividir más allá de 4: los grupos serían demasiado pequeños y confusos. |
| Validación cruzada | Repetir este análisis si se añaden nuevas features (p.ej. rating_mean). |

<div align="center">
  <img src="https://github.com/user-attachments/assets/e4b99a63-9f5a-4c10-87b1-2e79ec1f003f" alt="Clusters en R vs F" width="1280" height="720" />
  <div align="center">
    Clusters en R vs F
  </div>
</div>

### Gráfico 15. “Clusters en R vs F”

**Datos duros**

- Eje X: R_score estandarizado (recencia). Valores van de ≈ –1.5 (muy reciente) a +1.3 (poco reciente).
- Eje Y: F_score estandarizado (frecuencia). De ≈ –1.2 (baja frecuencia) a +1.7 (alta frecuencia).
- Cada color es un cluster (0 celeste, 1 naranja, 2 verde, 3 rojo).

**Sobre el gráfico**

- **Cluster 3 (rojo)**: puntos en la esquina superior derecha (R alto, F alto): clientes que compran muy seguido y recientemente → la “élite” de la actividad.
- **Cluster 2 (verde)**: abajo a la izquierda (R bajo, F bajo): clientes que no compran desde hace tiempo y además compran poco → grupo en riesgo.
- **Cluster 1 (naranja)**: arriba a la izquierda (R bajo frecuencia, pero muy recientes): compradores recientes pero no muy frecuentes.
- **Cluster 0 (celeste)**: frecuencia media-baja y recencia media, un grupo intermedio.

**Implicaciones del negocio**

| Cluster | Estrategia clave |
| --- | --- |
| 3 | “Power users”: programa VIP, early-access a lanzamientos y recompensas exclusivas. |
| 2 | “At risk”: campañas de re-enganche con ofertas agresivas y recordatorios (“Te extrañamos”). |
| 1 | “Recent but light”: sugerencias de cross-sell para aumentar su frecuencia de compra. |
| 0 | “Medio”: test de promociones A/B para ver qué incentiva más su recurrencia o ticket promedio. |

<div align="center">
  <img src="https://github.com/user-attachments/assets/615e9dc1-9db6-4383-9c7d-076adde412b2" alt="Monetary por cluster" width="1280" height="720" />
  <div align="center">
    Monetary por cluster
  </div>
</div>

### Gráfico 16. “Monetary por cluster”

**Datos duros**

- Cuatro cajas, una por cluster (0–3).
- Cluster 1: mediana ~$2,200 USD, rango IQR ~$1,900–$2,350.
- Cluster 3: mediana ~$2,000 USD, IQR ~$1,850–$2,300, con varios outliers arriba de $3,000.
- Cluster 0: mediana ~$1,500 USD, IQR ~$1,250–$1,850.
- Cluster 2: mediana ~$1,350 USD, IQR ~$1,100–$1,550.

**Sobre el gráfico**

- **Cluster 1** es el de **mayor gasto** promedio por cliente (perfil premium).
- **Cluster 2** gasta menos y presenta la caja más baja: clientes de bajo valor.
- **Clusters 0 y 3** están en niveles intermedios, aunque el 3 tiene más variabilidad y varios “big spenders”.

**Implicaciones del negocio**

| Cluster | Acción |
| --- | --- |
| 1 | Upsell de productos de alta gama; mantener stock premium y ofertas exclusivas. |
| 2 | Incentivos de bundle/paquetes para subir su ticket promedio; cross-sell de accesorios. |
| 3 | Segmentar dentro de 3: flow de fidelización para aquellos outliers de gasto ultra alto. |
| 0 | Promociones de valor medio; promociones para moverlos hacia el rango de gasto de Cluster 1. |

# 7. Visualización de segmentos (Fase 5)

### 7.1 Scatter plot de R_score vs F_score coloreado por cluster

```python
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
```

1. Partimos del DataFrame `rfm`, que ya incluye, para cada cliente, sus puntuaciones `R_score` (recencia) y `F_score`(frecuencia), y la etiqueta de `cluster` asignada por K-Means.
2. `plt.figure(figsize=(6,4))` abre un lienzo de 6×4 pulgadas para dibujar.
3. **Dibujar un punto por cliente**
    - El bucle `for c in sorted(rfm['cluster'].unique())` recorre cada cluster (0,1,2,3).
    - `subset = rfm[rfm['cluster']==c]` extrae sólo los clientes de ese cluster.
    - `plt.scatter(...)` dibuja un punto en el plano donde el eje X es `R_score` y el Y es `F_score`.
    - El parámetro `alpha=0.6` hace los puntos semi-transparentes, de modo que donde se amontonan se note más densidad.
4. **Etiquetas y leyenda**
    - `plt.xlabel` y `plt.ylabel` nombran los ejes.
    - `plt.title` añade el título arriba.
    - `plt.legend()` dibuja una cajita que asocia cada color con “Cluster 0”, “Cluster 1”, etc.
5. **Ajuste final y mostrar**
    - `plt.tight_layout()` ajusta márgenes para que nada se corte.
    - `plt.show()` presenta el gráfico.

> En conjunto, este scatter nos permite ver en qué zonas del espacio R–F (recencia vs frecuencia) se agrupan los clusters. Un cluster muy “arriba a la derecha” son clientes recientes y muy frecuentes; otro en “abajo a la izquierda” serían clientes con baja recencia y baja frecuencia.
> 

---

### 7.2 Boxplot de monetary (gasto) por cluster

```python
plt.figure(figsize=(6,4))
sns.boxplot(x='cluster', y='monetary', data=rfm)
plt.title('Distribución de gasto total (monetary) por cluster')
plt.xlabel('Cluster')
plt.ylabel('Gasto Total (USD)')
plt.tight_layout()
plt.show()
```

1. **Preparar gráfico**
    
    Igual que antes, abrimos un lienzo con `plt.figure`.
    
2. **Diagramas de caja (“boxplot”)**
    - `sns.boxplot` de Seaborn automatiza la creación de un diagrama de caja para cada cluster.
    - En el eje X aparecen los números de cluster (0, 1, 2, 3) y en el eje Y los valores de `monetary` (gasto total por cliente).
3. **Interpretación del boxplot**
    - La **caja** (box) muestra el rango intercuartílico (del 25 % al 75 %).
    - La **línea central** es la mediana de gasto.
    - Los **“bigotes”** extienden hasta 1.5×IQR; fuera de ellos, los puntos se consideran outliers (clientes con gasto excepcionalmente bajo o alto).
4. **Título y etiquetas**
    
    Nombramos el gráfico y los ejes igual que antes.
    
5. **Mostrar**
    
    `plt.show()` pinta todo.
    

> En conjunto, este boxplot revela cómo difiere el gasto total promedio entre clusters: por ejemplo, un cluster puede tener mediana de $150 USD y otro de $80 USD, lo que identifica claramente a los grupos de mayor valor económico.
> 

---

### 7.3 Tamaño de cada cluster (bar chart)

```python
plt.figure(figsize=(6,4))
rfm['cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('Número de clientes por cluster')
plt.xlabel('Cluster')
plt.ylabel('Cantidad de clientes')
plt.tight_layout()
plt.show()
```

1. **Contar clientes por cluster**
    - `rfm['cluster'].value_counts()` devuelve cuántos clientes hay en cada etiqueta de cluster, pero en orden descendente de frecuencia.
    - Con `.sort_index()` reordenamos por el número de cluster (0, 1, 2, 3).
2. **Dibujar barras**
    - `.plot(kind='bar')` crea un gráfico de barras verticales donde la altura refleja el número de clientes.
3. **Etiquetas y despliegue**
    - Añadimos título y nombres de ejes, ajustamos márgenes y mostramos con `plt.show()`.

> En conjunto, esta barra nos dice el tamaño de cada segmento descubierto por K-Means. Si “Cluster 2” tiene 60 clientes y “Cluster 0” sólo 15, sabremos dónde concentrar recursos (por ejemplo, diseñar ofertas de re-enganche para el cluster más numeroso).
> 

## Analizando el R_score vs. F_score, la distribución de gasto total y los números de clientes por cluster

<div align="center">
  <img src="https://github.com/user-attachments/assets/5f9b44ab-fc81-43cf-b4c0-cef04e187691" alt="Clusters R_score vs F_score" width="1280" height="720" />
  <div align="center">
    Clusters R_score vs F_score
  </div>
</div>

### Gráfico 17. “Clusters R_score vs F_score”

**Datos duros**

- Eje X: R_score (1–5), sin estandarizar.
- Eje Y: F_score (1–5).
- Cada punto coloreado por cluster.

**Sobre el gráfico**

Los clusters ocupan bloques de la cuadrícula R×F:

- **Cluster 3 (rojo)** en R=4–5 y F=3–5: clientes recientes y frecuentes.
- **Cluster 2 (verde)** en R=1–3 y F=1–2: clientes antiguos y poco frecuentes.
- **Cluster 1 (naranja)** en R=1–3 y F=3–5: recientes pero frecuencia media-alta.
- **Cluster 0 (celeste)**: frecuentes moderados o recientes con baja frecuencia.

**Implicaciones del negocio**

| Cluster | Insight R×F | Táctica recomendada |
| --- | --- | --- |
| 3 | Alto R, alta F | Recompensa VIP, upsell continuo |
| 2 | Bajo R, baja F | Reactivación intensiva, encuestas para entender abandono |
| 1 | Bajo R, alta F | Incentivos de recencia (p.ej. “compra antes de 30 días”) |
| 0 | Alto R, baja F | Campañas de frecuencia (“compra 2a visita con descuento”) |

<div align="center">
  <img src="https://github.com/user-attachments/assets/15b67553-fd7a-444b-8f5d-71b53044d7e8" alt="Distribución de gasto total (monetary) por cluster" width="1280" height="720" />
  <div align="center">
    Distribución de gasto total (monetary) por cluster
  </div>
</div>

### Gráfico 18. “Distribución de gasto total (monetary) por cluster”

**Datos duros**

Se ve igual que en el gráfico 16: medianas, IQR y outliers de gasto por cluster.

**Sobre el gráfico**

- Cluster 1 y 3 claramente por encima de 0 y 2 en gasto.
- Cluster 2 el más bajo, con outliers que no superan $2,000.

**Implicaciones del negocio**

- Mismas acciones sugeridas en el análisis de boxplot anterior: enfocar upsell en clusters altos, re-enganche en clusters bajos, y subdividir cluster 3 para identificar sus outliers de gasto ultra alto.

<div align="center">
  <img src="https://github.com/user-attachments/assets/251cf8cd-a28d-42e5-98ea-d449bc315e90" alt="Número de clientes por cluster" width="1280" height="720" />
  <div align="center">
    Número de clientes por cluster
  </div>
</div>

### Gráfico 19. “Número de clientes por cluster”

**Datos duros**

- **Cluster 0**: alrededor de **27 clientes**.
- **Cluster 1**: alrededor de **40 clientes**.
- **Cluster 2**: alrededor de **49 clientes**.
- **Cluster 3**: alrededor de **50 clientes**.

**Sobre el gráfico**

- **Cluster 3** es el más poblado (≈50), seguido muy de cerca por **Cluster 2** (≈49).
- **Cluster 1** es un grupo intermedio (≈40).
- **Cluster 0** es el más pequeño (≈27).
- Esto refleja que, al segmentar con K-Means en 4 grupos, la mayoría de clientes cae en dos clusters grandes con comportamientos distintos, mientras que un cluster es relativamente minoritario.

**Implicaciones del negocio**

| Cluster | Tamaño relativo | Acción principal |
| --- | --- | --- |
| 3 | Muy grande | Priorizar campañas de fidelización masiva: newsletters personalizadas y programas de puntos. |
| 2 | Muy grande | Lanzar ofertas de retención y upsell focalizadas para mantener su alto nivel de engagement. |
| 1 | Medio | Diseñar tests A/B de promociones para elevar su frecuencia o ticket promedio. |
| 0 | Pequeño | Implementar tácticas de reactivación individual (cupones especiales, llamadas de atención). |
- **Enfoque**: dedicar mayor presupuesto a los clusters 2 y 3 (mayor volumen), pero sin descuidar las estrategias de reactivación y upsell en los clusters más pequeños (0 y 1), donde pequeñas inversiones pueden generar saltos proporcionales grandes en ventas.

---

# 8. Recomendaciones por segmento y simulación de ROI

## Estrategias por segmento RFM

| Cluster | Tamaño (# clientes) | Gasto mediano (USD) | Estrategia clave |
| --- | --- | --- | --- |
| 3 (“Power users”) | 50 | 2 000 | Programa VIP y upsell continuo |
| 1 (“Premium”) | 40 | 2 200 | Upsell de alta gama; ofertas exclusivas |
| 0 (“Medio”) | 27 | 1 500 | Test A/B de promociones para elevar ticket |
| 2 (“At risk”) | 49 | 1 350 | Re-enganche con cupones agresivos |

### Supuestos para la simulación

- **Uplift esperado:** +10 % en gasto mediano por cliente.
- **Inversión de marketing:**
    - Clusters 0 & 2: 5 % del ingreso base de cada cluster.
    - Clusters 1 & 3: 7 % del ingreso base (mayor inversión en premium/VIP).
- **Ingreso base por cluster** = (# clientes) × (gasto mediano).
- **Incremento de ingreso** = Ingreso base × 10 %.
- **Coste inversión** = Ingreso base × (5 % ó 7 %).
- **ROI** = Incremento de ingreso / Coste inversión.

| Cluster | Ingreso base (USD) | Uplift 10 % (USD) | Inversión (USD) | ROI (%) |
| --- | --- | --- | --- | --- |
| 0 | 27 × 1 500 = 40 500 | 4 050 | 40 500 × 5 %= 2 025 | 4 050/2 025 ≈ 200 % |
| 1 | 40 × 2 200 = 88 000 | 8 800 | 88 000 × 7 %= 6 160 | 8 800/6 160 ≈ 143 % |
| 2 | 49 × 1 350 = 66 150 | 6 615 | 66 150 × 5 %= 3 307.5 | 6 615/3 307.5 ≈ 200 % |
| 3 | 50 × 2 000 = 100 000 | 10 000 | 100 000 × 7 %= 7 000 | 10 000/7 000 ≈ 143 % |
| **Total** | 294 650 | 29 465 | 18 492.5 | ≈ 159 % |

**Interpretación:**

- Clusters 0 y 2 ofrecen un ROI muy alto (200 %) con baja inversión (%).
- Clusters 1 y 3, siendo premium, requieren mayor inversión y entregan ROI ≈ 143 %.
- En conjunto, un uplift del 10 % genera ≈ 29 465 USD adicionales con un ROI global ≈ 159 %.

---

## Bundles de producto para ticket medio

**Insight:** pico de compras entre $40–$60 USD .

**Acción propuesta:**

- Diseñar “bundle básico” (por ejemplo, camiseta + accesorio) que se ofrezca automáticamente al llegar al carrito con valor $40.
- Incentivo: 10 % de descuento en el bundle.

**Simulación de impacto:**

- Supongamos que el bundle impulsa un 10 % de clientes de ticket medio (actualmente 60 % de transacciones) a gastar +$8 adicionales.
- Si el ticket medio actual de ese segmento es $50 y hay 1 000 transacciones/mes:
    - Incremento por transacción: $8 × 100 (10 % de 1 000) = $800
    - Inversión en descuentos: $8 × 100 × 10 % = $80
    - ROI ≈ $800/80 = 1 000 %

---

## Programa de fidelidad y referidos

**Insight:** alta frecuencia media (20–24 compras) sugiere compradores recurrentes .

**Acción propuesta:**

- Crear programa de puntos: 1 punto por cada $10 gastados; canjeable por cupones.
- Incentivar referidos: $5 de crédito por cada amigo que compre.

**Supuestos y simulación:**

- Si el 15 % de clientes recurrentes (≈ 500 clientes/mes) generan 2 referidos y estos gastan $30 de media:
    - Nuevos ingresos = 500 × 2 × 30 = $30 000
    - Coste de referidos = 500 × 2 × 5 = $5 000
    - ROI = $30 000/5 000 = 600 %

---

## Re-enganche de clientes “At risk”

**Insight:** Cluster 2 (“At risk”) con 49 clientes y gasto mediano $1 350 .

**Acción propuesta:**

- Cupón de re-enganche: “Te extrañamos: 15 % off en tu próxima compra”.
- Envío de email/SMS personalizado.

**Simulación:**

- Supongamos que el 20 % de estos clientes (≈ 10) responden y gastan de nuevo $1 350:
    - Ingreso recuperado = 10 × 1 350 = $13 500
    - Coste de cupón = 13 500 × 15 % = $2 025
    - ROI = 13 500/2 025 ≈ 667 %

---

## Incentivos de pago digital

**Insight:** 48 % cash vs 52 % tarjeta .

**Acción propuesta:**

- “Cashback” del 5 % al pagar con tarjeta.
- Comunicación de beneficios (seguridad, puntos).

**Simulación:**

- Si de 1 000 transacciones de efectivo, el 10 % (100) migra a tarjeta y el ticket medio sube de $80 a $88 (10 % uplift):
    - Incremento ingreso = 100 × 8 = $800
    - Coste cashback = 100 × 88 × 5 % = $440
    - ROI = 800/440 ≈ 182 %

---

## Promociones estacionales y días “flojos”

| Día/Mes | Insight | Acción | Simulación 10 % uplift |
| --- | --- | --- | --- |
| Martes | Ventas más bajas | “Happy Tuesday”: 10 % off exclusivo | +10 % ventas martes → +$1 200/mes; ROI ≈ 250 %¹ |
| Mes valle (ej. junio) | Ventas < $15 000 | “Mid-season sale” | +10 % ventas mensuales → +$1 500; ROI ≈ 300 %² |

¹ Suponiendo ventas martes = $12 000/mes, uplift $1 200, inversión $480 (40 % de uplift).

² Ventas valle = $15 000, uplift $1 500, inversión $500 (33 % de uplift).

---

## Conclusión

- **Segmentación RFM** ofrece el mayor detalle para asignar presupuesto y predecir ROI (véase tabla de § 6.1).
- **Bundles y fidelidad** son palancas de alto ROI (> 600 %).
- **Re-enganche** y **promociones en días valle** permiten recuperar o suavizar caídas con inversiones muy bajas.
- **Incentivos de pago digital** aceleran adopción y elevan ticket medio con ROI > 180 %.

---

<p align="center">
<big><strong>✨¡Gracias por visitar este repositorio!✨</strong></big>
</p>
