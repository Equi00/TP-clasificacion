# En este trabajo practico se analizara un dataset que indica informacion importante sobre enfermedades cardiacas.
# Las tablas que tenemos en nuestro dataset son: age, sex, cp(tipo de dolor de pecho), trestbps (presion arterial en
# reposo), chol (colesterol serico), fbs(glucemia en ayunas), restecg (electrocardiograma en reposo),
# thalach (frecuencia cardiaca maxima alcanzada), exang (angina inducida por ejercicio), oldpeak (Depresión del ST
# inducida por el ejercicio en relación con el reposo), slope (la pendiente del segmento ST de ejercicio máximo),
# ca (número de vasos principales), thal y target.
# Tenemos como objetivo realizar predicciones de la variable "target" utilizando las variables que mejor esten
# correlacionadas con la misma.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/heart.csv")

print("\nDESCRIPCION DE LA TABLA")
print(df.describe())

print("\nCOLUMNAS")
print(df.columns)

print("\nPRIMEROS DATOS DE LA TABLA")
print(df.head())

print("\nINFORMACION DE LA TABLA")
print(df.info())

print("\nCANTIDAD DE VALORES NULL POR COLUMNA")
print(df.isnull().sum())

# Podemos observar que en nuestro dataset tenemos un total de 303 registros y unas 14 columnas, no tenemos valores nulos
# en ninguna columna. Tambien es importante ver que no tenemos datos categoricos, por lo que no hara falta hacer alguna
# conversion de datos.

# Vamos a analizar con un mayor enfasis la descripcion de nuestras columnas

print("\nDESCRIPCION DE LAS COLUMNAS age, sex Y cp")
print(df[['age', 'sex', 'cp']].describe())

# La edad media de las personas registradas en el dataset es de 54 anios, por lo que podemos suponer que la mayoria de
# las personas con problemas cardiacos rondan por los 50 anios, si bien no son personas viejas estos problemas pueden
# deberse a otros factores como los habitos, estilo de vida o la genetica. Tambien podemos ver que las personas mas
# jovenes con estos problemas tiene 29 anios y las personas mas mayores registradas en esta tabla tienen 77 anios.

# La columna de sexo esta compuesta por unos y ceros, 1 para hombres y 0 para mujeres, el promedio de todos estos es de
# casi 0.7, por lo que la mayoria de personas registradas en el dataset son hombres. Esto nos puede decir que los
# hombres son mas propensos a sufrir problemas cardiacos que las mujeres, esto quiza pueda deberse a una condicion
# biologica o a las actividades/trabajos que implican grandes cantidades de energia y sobreesfuerzo,
# aun asi no tenemos los suficientes registros como para poder sacar esta conclusion con exactitud.

# La columna de cp (tipo de dolor de pecho) esta compuesta de 4 valores: 0 angina tipica, 1 angina atipica,
# 2 dolor no anginoso, 3 asintomatico. Por lo visto el valor medio es de casi 1, la mayoria de las personas en el
# dataset tienen angina atipica, un dolor de pecho que no se ajusta con los patrones de la angina tipica.

print("\nDESCRIPCION DE LAS COLUMNAS trestbps, chol Y fbs")
print(df[['trestbps', 'chol', 'fbs']].describe())

# En la columna trestbps (presion arterial en reposo) esta medido en milimetros de mercurio (mm/hg), se puede observar
# que la media de las personas diagnosticadas tienen un valor de 131 aproximadamente, esto nos dice que la mayoria de
# los pacientes tienen una presion arterial levemente mayor a la normal (120 mm/hg), aun asi se encuentra dentro del
# rango aceptable. El valor mas bajo de presion arterial es de 94, este valor de presion representa hipotension (presion
# arterial baja) y el valor maximo es de 200, este valor de presion representa hipertension (presion arterial alta).

# En la columna chol (colesterol serico) esta medido en miligramos por decilitro (mg/dl), se puede observar que la media
# es de 246 aproximadamente, lo que representa que la mayoria de los pacientes tienen una cantidad de colesterol por
# encima de los niveles normales que seria menos de 200 mg/dl. Esto puede deberse a casos en que la genetica de algunas
# personas las haga mas propensas al colesterol en sangre, la mala alimentacion de las personas, la inactividad fisica,
# la obesidad o por alguna enfermedad como la diabetes. Vemos que el valor minimo es de 126, esto esta dentro del valor
# normal, y el valor maximo es de 564 lo que indica un valor extremadamente elevado.

# En la columna fbs (glucemia en ayunas mayor a 120 mg/dl) vemos que esta compuesta por unos y ceros, 1 si, 0 no.
# Podemos ver que la media es de casi 0, por lo que la mayoria de las personas tienen una glucemia en ayunas menor o
# igual a 120 mg/dl. Los valores normales estan entre 70 y 100 mg/dl, entonces la mayoria de los pacientes del dataset
# no representan signos de tener diabetes.

print("\nDESCRIPCION DE LAS COLUMNAS restecg, thalach Y exang")
print(df[['restecg', 'thalach', 'exang']].describe())

# En la columna restecg (electrocardiograma en reposo) hay 3 valores: 0: mostrando hipertrofia ventricular izquierda probable o
# definitiva según los criterios de Estes, 1: normal, 2: con anomalía de la onda ST-T. De los cuales la media es de
# 0.5 aproximadamente, podemos suponer que una gran cantidad de pacientes tienen un valor normal y otra gran cantidad
# tienen hipertrofia ventricular izquierda probable o definitiva.

# En la columna thalach (frecuencia cardiaca maxima alcanzada) vemos que el valor medio es de 149 aproximadamente, segun
# una estimacion general, este valor se puede sacar restando 220 la edad del paciente. Entonces este valor medio nos
# esta diciendo que la mayoria de personas del dataset tienen 220 - 149 = 71 anios, pero esto no tiene sentido ya que
# en la columna de la edad figura que la mayoria de personas tienen unos 50 anios aproximadamente. Esto quiere decir que
# la mayoria de las personas tienen una frecuencia cardiaca maxima bastante baja, esto se debe a las enfermedades
# cardiacas de los pacientes, esta frecuencia tambien representa que las personas no pueden realizar mucha actividad
# fisica.
# El valor minimo es de 71, lo cual para una persona sana es imposible de alcanzar, representa un grave problema
# cardiaco. El valor maximo es de 202, esto puede ser un valor saludable, puede deberse a un buen estado fisico, por
# otro lado esto tambien puede deberse a estar bajo grandes cantidades de estress o ansiedad.

# En la tabla de exang (angina inducida por ejercicio) tenemos los valores como unos y ceros, 1 si, 0 no. El valor medio
# es de 0.3 aproximadamente, esto indica que la mayoria de los pacientes del dataset no tienen angina inducida por
# ejercicio.

print("\nDESCRIPCION DE LAS COLUMNAS oldpeak, slope Y ca")
print(df[['oldpeak', 'slope', 'ca']].describe())

# En la columna oldpeak (Depresión del ST inducida por el ejercicio en relación con el reposo) vemos que la media indica
# un valor de 1, teniendo en cuenta que esto se expresa en milimetros, la mayoria de gente esta por dentro de los
# valores normales que son entre 0 y 1 mm. Vemos que el valor minimo es de 0 mm y el valor maximo es de 6.2 mm, este
# indicador puede significar que existe una disfuncion coronaria significativa.

# En la columna slope (la pendiente del segmento ST de ejercicio máximo) tiene 3 valores: 0: pendiente creciente;
# 1: plano; 2: pendiente decreciente. Vemos que la media indica un valor de 1.4, la mayoria de los pacientes tienen
# una pendiente plana o decreciente. Tener una pendiente plana significa falta de respuesta o isquemia miocardiaca
# subyacente, y una pendiente decreciente se considera un hallazgo anormal y puede ser un indicador de una disfunción
# coronaria significativa o una isquemia miocárdica.

# En la columna ca (número de vasos principales) tenemos valores que van del 0 al 3. Por lo visto en esta columna nos
# indica que tenemos como maximo un numero 4, tenemos datos mal puestos en esta columna. Procedemos a eliminar estos
# datos.

df["ca"] = df["ca"].replace(4, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()


def drop_na(data):
    data = data.dropna(axis=0)

    print("\nCANTIDAD DE VALORES NULL POR COLUMNA")
    print(data.isnull().sum())

    sns.heatmap(data.isnull(), cbar=False)
    plt.show()

    return data


df = drop_na(df)

# Una vez eliminados los valores erroneos volvemos a ver la descripcion de la columna ca.

print("\nDESCRIPCION DE LA COLUMNA ca SIN VALORES 4")
print(df["ca"].describe())

# Observamos que se eliminaron un total de 5 registros, lo que es muy bueno para futuros analisis. Vemos que el valor
# medio es de casi 0.7, indica que la mayoria de personas en el dataset tinene un CA1, tienen una arteria coronaria
# principal muestra estenosis o estrechamiento significativo. esto puede ser un factor de riesgo para problemas
# cardíacos, como el infarto de miocardio.

print("\nDESCRIPCION DE LAS COLUMNAS thal y target")
print(df[['thal', 'target']].describe())

# En la columna de thal tenemos 3 valores: 1 = defecto fijo, 2 = defecto reversible, 3 = normal. Se puede observar que
# el valor minimo es de 0, por lo que otra vez tenemos datos erroneos en el dataset, procedemos a eliminarlos.

df["thal"] = df["thal"].replace(0, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()

df = drop_na(df)

# Una vez eliminados los valores null volvemos a analizar las columnas.

print("\nDESCRIPCION DE LAS COLUMNAS thal y target")
print(df[['thal', 'target']].describe())

# Podemos ve que se eliminaron solo 2 registros del dataset, este resultado es mas que aceptable. Vemos que el valor
# promedio de thal es de 2.3, esto indica que la mayoria de los pacientes tienen un defecto reversible. Significa que
# hay áreas en el corazón donde el flujo sanguíneo está reducido durante el estrés (como durante una prueba de
# esfuerzo) pero se restablece cuando el estrés se detiene. Esto puede ser indicativo de una isquemia miocárdica
# inducible, lo que significa que hay áreas del corazón que no reciben suficiente flujo sanguíneo durante la
# actividad física o el estrés, pero que se recuperan cuando el estrés se alivia.

# En la columna target tenemos 2 valores, 0 sin enfermedad, 1 con enfermedad. El promedio esta en 0.54, osea que un
# poco mas de la mitad de las personas del dataset sufren de alguna enfermedad cardiaca. Por lo que la otra mitad que
# sufre de dolores o sintomas extranios se deben a otros motivos.

# Gráfico de distribución para cada variable numérica
# ==============================================================================

fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(9, 5))
axes = axes.flat

columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data=df,
        x=colum,
        stat="count",
        kde=True,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        line_kws={'linewidth': 2},
        alpha=0.3,
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=8)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución de variables numéricas', fontsize=10, fontweight="bold")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(12, 6))
columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

axs = axs.flatten()

for i, colum in enumerate(columnas_numeric):
    axs[i].hist(df[colum], bins=20, color="#3182bd", alpha=0.5)
    axs[i].plot(df[colum], np.full_like(df[colum], -0.01), '|k', markeredgewidth=1)
    axs[i].set_title(f'{colum}')
    axs[i].set_xlabel(colum)
    axs[i].set_ylabel('counts')

plt.tight_layout()
plt.show()

# Podemos observar de froma grafica que nuestros analisis previos de los datos eran acertados en la mayoria.

# la edad de la mayoria de los pacientes ronda entre los 50 y 60 anios, la mayoria de estos son hombres,
# podemos observar que en cp (dolor de pecho) la mayoria de las personas tienen angina tipica siguiendole en gran
# cantidad las personas con dolor no anginoso, con esta distribucion parecia que la mayoria de las personas tenian
# angina atipica por los resultados del promedio. en el grafico de trestbps (presion arterial en reposo) la mayoria
# tiene una precion arterial de 130. en el grafico de chol (colesterol serico) la mayoria tiene un colesteron de
# entre 240 y 250. En el grafico de fbs (glucemia en ayunas mayor a 120 mg/dl) la mayoria no precenta tener una
# cantidad de glucemia mayor a 120. En el grafico de restecg (electrocardiograma en reposo) la mayoria esta normal o
# con hipertrofia ventricular izquierda probable o definitiva. En el grafico de thalach (frecuencia cardiaca maxima
# alcanzada) el promedio es de 150 aproximadamente. En el grafico de exang (angina inducida por ejercicio) la mayoria
# de pacientes no tienen angina inducida durante el ejercicio. En el grafico de oldpeak (Depresión del ST inducida
# por el ejercicio en relación con el reposo) la mayoria de gente esta dentro del rango normal. En el grafico de
# slope (la pendiente del segmento ST de ejercicio máximo) la mayoria tiene una pendiente horizontal o una pendiente
# ascendente. En el grafico ca (número de vasos principales) vemos que la mayoria esta en CA0, no se observa
# estenosis significativa en ninguna de las arterias coronarias principales, a diferencia de lo que habiamos supuesto
# en el previo analisis. En el grafico de thal se puede observar que la mayoria tiene un defecto reversible,
# pero tambien se ve que una gran parte esta normal. En el grafico de target vemos que la cantidad de gente con
# enfermedad es mayoritaria por poco.

# Análisis Univariado

# Medidas de centralización: media, mediana y moda
columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns
for i, colum in enumerate(columnas_numeric):
    print(f"\nMEDIDAS DE CENTRALIZACION {colum}")
    print(f'Media:{df[colum].mean()} \
     \nMediana: {df[colum].median()} \
     \nModa: {df[colum].mode()}')

# En age tenemos MEDIA < MEDIANA < MODA, esto significa que hay distribucion sesgada negativamente de los datos.
# En sex tenemos MEDIA < MEDIANA = MODA, esto significa que hay districucion sesgada negativamente.
# En cp tenemos MEDIA = MEDIANA > MODA, esto significa que hay distribucion sesgada positivamente.
# En trestbps tenemos MEDIA > MEDIANA > MODA, esto significa que hay distribucion sesgada positivamente.
# En chol tenemos MEDIA > MEDIANA > MODA, esto significa que hay distribucion sesgada positivamente.
# En fbs tenemos MEDIA > MEDIANA = MODA, esto significa que hay distribucion sesgada positivamente.
# En restecg tenemos MEDIA < MEDIANA = MODA, esto significa que hay districucion sesgada negativamente.
# En thalach tenemos MEDIA < MEDIANA < MODA,esto significa que hay distribucion sesgada negativamente.
# En exang tenemos MEDIA > MEDIANA = MODA, esto significa que hay distribucion sesgada positivamente.
# En oldpeak tenemos MEDIA > MEDIANA > MODA, esto significa que hay distribucion sesgada positivamente.
# En slope tenemos MEDIA > MEDIANA < MODA y MODA > MEDIA, esto significa que hay districucion sesgada negativamente.
# En ca tenemos MEDIA > MEDIANA = MODA, esto significa que hay distribucion sesgada positivamente.
# En thal tenemos MEDIA > MEDIANA = MODA, esto significa que hay distribucion sesgada positivamente.
# En target tenemos MEDIA < MEDIANA = MODA, esto significa que hay districucion sesgada negativamente.

# Medidas de dispersión: desviación típica, rango, IQR, coeficiente de variación, desviación media

print(f'\nLa varianza es:\n{df.var()}')

# la varianza de las columnas sex, cp, fbs, restecg, exang, oldpeak, slope, ca, thal y target tienen una varianza baja,
# por lo que su dispecion de los datos tambien es baja, tambien se debe a que sus datos se refieren a un tipo como
# la columna de sex, donde solo hay unos y ceros.
# la varianza de las columnas age, trestbps, chol y thalach son altas, por lo qeu su dispercion de los datos es alta.

print(f'\nDesviación Estándar por fila:\n{df.std(axis=0)}')

for i, colum in enumerate(columnas_numeric):
    print(f"\nRANGO DE {colum}")
    print(f'El rango es: {df[colum].max() - df[colum].min()}')

for i, colum in enumerate(columnas_numeric):
    print(f"\nEL RANGO INTERCUATRILICO DE {colum}")
    print(f'El IQR es: {df[colum].quantile(0.75) - df[colum].quantile(0.25)}')

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
df_varianza = df.apply(cv)
print(
    f'\nEl coeficiente de variación es:\n{df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"]).apply(cv)}')

# Podemos comprobar lo anteriormente dicho observando los valores de la desviacion estandar por fila, los rangos de cada
# columna, el rango intercuatrilico de las columnas y el coeficiente de variacion.

# Medidas de asimetría

print(f"\nLas medidas de asimetría son:\n{df.skew()}")

# Las medidas de asimetria nos muestra que todas las columnas tienen un valor muy cercano a cero, con pequenias
# tendencias hacia una asimetria negativa o positiva.

print(f"\nLas medidas de kurtosis son:\n{df.kurt()}")

# Segun las medidas de kurtosis, las columnas age, sex, cp, restecg, thalach, exang, slope, thal y target tienen una
# distribucion platicurtica. el resto tiene una distribucion leptocurtica, siendo mas notable en chol y fbs. La mayoria
# de las columnas tienen un valor muy cercano a cero, lo que daria una distribucion mesocurtica.

# Ahora que ya analizamos todos nuestros datos vamos a seguir con la deteccion y eliminacion de outliers

df.plot(kind='box', subplots=True, layout=(2, 7),
        sharex=False, sharey=False, figsize=(20, 10))
plt.show()


# Se observa la precensia de varios outliers en multiples columnas, por suerte la mayoria de estas no precentan tener
# outliers.

def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr

    ls = df.index[(df[ft] < low) | (df[ft] > up)]

    return ls


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


def delete_outliers(n, df, list):
    df_clean = df
    for i in range(n):
        index_list = []
        for feature in list:
            index_list.extend(outliers(df_clean, feature))
        if not index_list:
            break
        df_clean = remove(df_clean, index_list)
    return df_clean


df_sin_outlier = delete_outliers(100, df, columnas_numeric)

print("\nTABLA SIN LOS OUTLIERS")
print(df_sin_outlier.shape)
print(df_sin_outlier.info())

df_sin_outlier.plot(kind='box', subplots=True, layout=(2, 7),
                    sharex=False, sharey=False, figsize=(20, 10))
plt.show()

# La eliminacion de outliers fue exitosa, tenemos un total de 226 registros de 303 que teniamos en un principio. Todavia
# seguimos teniendo la mayoria de los datos, sin embargo, esta eliminacion podria darnos resultados erroneos en nuestros
# proximos analisis. Pero no podriamos realizar un dataset con los outliers reemplazados, ya que muchas de estas
# columnas tienen valores categoricos convertidos a numericos, ponerle la media a estos outliers entorpecerian los
# analisis.

# Ahora tenemos 2 datasets, una con todos los outliers y otra con los outliers eliminados.

# A continuacion analizaremos graficamente todas las columnas del dataset sin outliers, pero tenemos varias columnas
# que debemos cambiar a categorica si queremos tener un mejor entendimiento de sus valores.

df_categorico = df_sin_outlier.copy()

df_categorico['target'] = df_categorico.target.replace({1: "Con_enfermedad", 0: "Sin_enfermedad"})
df_categorico['sex'] = df_categorico.sex.replace({1: "Hombre", 0: "Mujer"})
df_categorico['cp'] = df_categorico.cp.replace(
    {0: "Angina_tipica", 1: "Angina_atipica", 2: "Dolor_no_anginoso", 3: "Asintomatico"})
df_categorico['exang'] = df_categorico.exang.replace({1: "Si", 0: "No"})
df_categorico['fbs'] = df_categorico.fbs.replace({1: "Verdadero", 0: "Falso"})
df_categorico['slope'] = df_categorico.slope.replace({0: "Creciente", 1: "Plano", 2: "Decreciente"})
df_categorico['thal'] = df_categorico.thal.replace({1: "Defecto_fijo", 2: "Defecto_reversible", 3: "Normal"})


# Una vez obtenemos nuestro dataset categorico, realizamos los graficos. Ademas de ver los graficos de cada columna
# tambien queremos ver los porcentajes de nuestras variables categoricas para tener una visualizacion mas entendible.

def grafico_porcentaje(ax):
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_x(), i.get_height() - 5,
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
                color='white', weight='bold')

    plt.tight_layout()
    plt.show()


fig, ax = plt.subplots(figsize=(5, 4))
name = ["Con_enfermedad", "Sin_enfermedad"]
ax = df_categorico.target.value_counts().plot(kind='bar')
ax.set_title("Enfermedad cardiaca", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

# Calculamos los porcentajes para despues ingresarlos en el grafico
grafico_porcentaje(ax)

# Podemos ver que hay una mayoria de gente con enfermedad cardiaca, superando en un 16% a la cantidad de gente sana.

fig, ax = plt.subplots(figsize=(8, 5))
name = ["Hombre", "Mujer"]
ax = sns.countplot(x='sex', hue='target', data=df_categorico, palette='Set2')
ax.set_title("Distribucion del genero segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# Se observa que hay mas hombres con enfermedad cardiaca que mujeres. Tambien se ve que hay una mayoria de hombres del
# dataset que no presentan enfermedades cardiacas, mientras tanto la mayoria de las mujeres del dataset si precentan
# tener alguna enfermedad cardiaca, solo el 5.31% de mujeres no tienen enfermedad.

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Dolor_no_anginoso", "Angina_atipica", "Angina_tipica", "Asintomatico"]
ax = sns.countplot(x='cp', hue='target', data=df_categorico, palette='Set2')
ax.set_title("Distribucion del dolor de pecho segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# Viendo el grafico de los dolores de pecho, de los asintomaticos tenemos que la mayoria precenta tener enfermedad
# cardiaca, de los que tienen angina atipica la gran mayoria tiene enfermedad cardiaca, los que tienen dolor no anginoso
# tambien precentan un gran porcentaje de gente con enfermedad cardiaca. Es curioso ver que la gente con
# angina tipica, la gran mayoria sean personas sanas. El dolor de pecho puede ser algo subjetivo o psicologico
# en algunos casos, como por ejemplo: al tener exceso de estres, al hacer actividad fisica, etc.

sns.countplot(x='fbs', hue='target', data=df_categorico, palette='Set2').set_title(
    'Distribucion de glucosa en sangre segun el target')
plt.show()

# A la hora de analizar la distribucion de glucosa en sangre segun el target (fbs), nos encontramos que solo estan los
# datos de "Falso", eso quiere decir que entre los outliers que se eliminaron, tambien se eliminaron los registros en
# donde este dato era "Verdadero". Lo que haremos es agarrar el dataset con outliers y analizar este caso en especifico.

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Falso", "Verdadero"]
ax = sns.countplot(x='fbs', hue='target', data=df, palette='Set2')
ax.set_title("Distribucion de glucosa en sangre > 120mg/dl segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# Ahora si podemos visualizar bien los datos, tenemos que muy poca gente supera el 120mg/dl, la cantidad de gente con
# enfermedad y sin enfermedad esta bastante nivelada. En cambio las personas que estan por debajo de los 120mg/dl, la
# mayoria precentan enfermedad cardiaca, superando a la gente sana en un 17% aproximadamente. Esto puede decirnos que
# la diabetes no es un factor de gran importancia cuando se quiere hablar de problemas cardiacos. La mayoria de gente
# que no precenta diabetes tiene un mayor porcentaje de precentar enfermedades cardiacas en el dataset.

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Creciente", "Plano", "Decreciente"]
ax = sns.countplot(x='slope', hue='target', data=df_categorico, palette='Set2')
ax.set_title("Distribucion del slope segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# De este grafico podemos observar que muy pocos pacientes tienen un slope (la pendiente del segmento ST de ejercicio
# máximo) creciente, que seria lo normal. La gran mayoria tiene un slote plano, de los cuales muchisimos tienen una
# enfermedad cardiaca, teniendo un 37.61% en comparacion con las personas sanas de slote plano 12.39%. Una gran parte de
# los pacientes tienen un slote decreciente, sin embargo la mayoria de estos no precentan enfermedad cardiaca.

fig, ax = plt.subplots(figsize=(10, 5))
name = ["No", "Si"]
ax = sns.countplot(x='exang', hue='target', data=df_categorico, palette='Set2')
ax.set_title("Distribucion de la angina inducida por ejercicio (exang) segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# Vemos que hay una gran cantidad de gente que no tiene angina inducida por ejercicio, sin embargo, de todos estos
# la inmensa mayoria precentan enfermedad cardiaca 49.12%, mientras que las personas sanas estan por el 19.47%. De las
# personas que tienen angina por ejercicio, la mayoria estan sin enfermedad, esto quiere decir que los dolores de pecho
# durante el ejercicio no esta relacionado a un problema del corazon, puede deberse a otros factores.

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Defecto_reversible", "Defecto_fijo", "Normal"]
ax = sns.countplot(x='thal', hue='target', data=df_categorico, palette='Set2')
ax.set_title("Distribucion del slope segun el target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

grafico_porcentaje(ax)

# En este grafico podemos determinar que la mayoria de los pacientes del dataset tienen un defecto reversible, pero la
# inmensa mayoria de estos precentan una enfermedad cardiaca en un 47.79%, mientras que los sanos son apenas unos 11.06%
# Hay muy poca gente con un defecto fijo y la cantidad de estos con o sin enfermedad es muy nivelada.
# De la gente que esta normal la gran mayoria no precentan enfermedades cardiacas.

# a continuacion veremos las correlaciones entre nuestras columnas.

# Heatmap matriz de correlaciones

print("\nMATRIZ DE CORRELACIONES CON OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nMATRIZ DE CORRELACIONES SIN OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_sin_outlier.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()


# En los graficos se ve que hay una buena correlacion positiva de las columnas cp, thalach, slope con target.
# Las columnas oldpeak, exang, ca, thal, sex, age tienen una buena correlacion negativa con target.
# El resto de columnas no tienen una buena correlacion con target.

# Ahora que logramos analizar todo nuestro dataset, vamos a empezar con nuestro analisis estadistico.
# Nuestro primer modelo sera la regresion estadistica lineal, primero separemos los datos.

def logistic(df, classifier):
    X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Predicción', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Matriz de Confusión', fontsize=17)
    plt.show()

    print(f'\nPrecisión del modelo: {precision_score(y_test, y_pred):.2f}')
    print(f'Exactitud del modelo:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Sensibilidad del modelo: {recall_score(y_test, y_pred):.2f}')
    print(f'Puntaje F1 del modelo:{f1_score(y_test, y_pred):.2f}')
    print(f'Curva ROC - AUC del modelo:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))


# Como tenemos multiples variables en X, graficarlo seria muy complicado, por lo que a fines practicos graficaremos
# unicamente a las variables que tengan la mayor correlacion con la columna target, pero no deben ser categoricas,
# asi podemos obtener un mejor grafico, en ambos dataset tenemos que las variables con mas alta correlacion no
# categoricas son 'age' y 'oldpeak'

def grafico_logistic(df, classifier):
    X = df[["age", "oldpeak"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Predicción', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Matriz de Confusión', fontsize=17)
    plt.show()

    print(f'\nPrecisión del modelo: {precision_score(y_test, y_pred):.2f}')
    print(f'Exactitud del modelo:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Sensibilidad del modelo: {recall_score(y_test, y_pred):.2f}')
    print(f'Puntaje F1 del modelo:{f1_score(y_test, y_pred):.2f}')
    print(f'Curva ROC - AUC del modelo:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Clasificador (Conjunto de Entrenamiento)')
    plt.xlabel('Edad')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Clasificador (Conjunto de Test)')
    plt.xlabel('Edad')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()


def mejor_k(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    media_pre = np.zeros((9))
    for n in range(1, 10):
        classifier = KNeighborsClassifier(n_neighbors=n, metric="minkowski", p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        media_pre[n - 1] = accuracy_score(y_test, y_pred)

    print(f"Mejor precision: {media_pre.max()}, con K = {media_pre.argmax() + 1}")


classifier = LogisticRegression(random_state=0)
print("\nRegresion logistica lineal para dataset normal:")
logistic(df, classifier)

# En el grafico de confusion del dataset con outliers vemos que tenemos un total de 5 falsos negativos y 7 falsos
# positivos, en comparacion con los verdaderos positivos y negativos es muy poco, por lo que las predicciones con el
# dataset con outliers es bastante buena. En el modelo tenemos una precicion del 81% y una exactitud del 84%, por lo
# tanto nuestro modelo puede sacar buenas predicciones.

print("\nRegresion logistica lineal para grafico de dataset normal:")
grafico_logistic(df, classifier)

# En el grafico de confusion observamos 16 falsos positivos y 6 falsos negativos, estos son muchas predicciones malas,
# por lo que este modelo es menos preciso a comparacion de utilizar las 9 columnas mas correlacionadas con target.
# Este modelo tiene una precision del 64% y una exactitud del 70%, nos deja malas predicciones, por lo que no es
# recomendable utilizarlo. Graficamente se ven muchos puntos verdes dentro del rango rojo y viceversa.

print("\nRegresion logistica lineal para dataset sin outliers:")
logistic(df_sin_outlier, classifier)

# En el grafico de confusion del dataset sin outliers vemos un total de 9 falsos positivos y 5 falsos negativos, por lo
# que tenemos mas predicciones malas en comparacion con el dataset con outliers. El modelo tiene una precision del 72%
# y una exactitud del 75%, si bien el modelo tiene mas predicciones buenas que malas, no es muy preciso y tambien es
# menos preciso que el dataset con outliers. Esto puede deberse a la poca cantidad de datos del dataset y que quiza
# los registros eliminados por los outliers eran bastante importantes para el analisis.

print("\nRegresion logistica lineal para grafico de dataset sin outliers:")
grafico_logistic(df_sin_outlier, classifier)

# En el grafico de confusion vemos que la cantidad de falsos positivos es muy alta con un numero de 18 y los falsos
# engativos es de 7. Podemos observar a simple vista que este modelo no es para nada preciso. Tenemos una precision del
# 54% y una exactitud del 56%, este modelo es muy malo para darnos alguna prediccion buena, en mucho menos exacto en
# comparacion con el dataset con outliers. Se ve una gran cantidad de puntos rojos en la zona verde y viseversa.

# El siguiente metodo que realizaremos es la regresion logistica utilizando KNN. Primero debemos encontrar el mejor K
# de cada dataset

X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
y = df["target"].values

print("\nMejor K general para dataset normal:")
mejor_k(X, y)

X = df[["age", "oldpeak"]].values
y = df["target"].values

print("\nMejor K para el grafico de dataset normal:")
mejor_k(X, y)

X = df_sin_outlier[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
y = df_sin_outlier["target"].values

print("\nMejor K general para dataset sin outliers:")
mejor_k(X, y)

X = df_sin_outlier[["age", "oldpeak"]].values
y = df_sin_outlier["target"].values

print("\nMejor K para el grafico de dataset sin outliers:")
mejor_k(X, y)

# una vez encontrado el mejor K procedemos a realizar la clasificacion

classifier_df_normal = KNeighborsClassifier(n_neighbors=9, metric="minkowski", p=2)
classifier_df_normal_grafico = KNeighborsClassifier(n_neighbors=6, metric="minkowski", p=2)
classifier_df_sin_outlier = KNeighborsClassifier(n_neighbors=4, metric="minkowski", p=2)
classifier_df_sin_outlier_grafico = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)

print("\nRegresion logistica KNN para dataset normal:")
logistic(df, classifier_df_normal)

# Vemos que KNN en el dataset normal nos da una precision de 82% y una exactitud de 88%, este modelo nos puede dar muy
# buenas predicciones, la cantidad de falsos positivos y falsos negativos es muy baja.

print("\nGrafico logistica KNN para dataset normal:")
grafico_logistic(df, classifier_df_normal_grafico)

# Por parte del grafico hay una cantidad de falsos negativos y falsos positivos un poco mas grande, el modelo no va a
# ser tan preciso. Tenemos un total de precision del 72% y una exactitud del 74%, no es un porcentaje muy malo pero
# el hecho de tener un casi 30% de fallas hace que este modelo no sea recomendable. En el grafico tampoco convence
# mucho.

print("\nRegresion logistica KNN para dataset sin outlier:")
logistic(df_sin_outlier, classifier_df_sin_outlier)

# Podemos ver una cantidad de falsos negativos y falsos positivos bastante baja en el dataset sin outliers, esto nos
# dice que vamos a tener una buena precision en el modelo, sin embargo hay que tener en cuenta que este dataset tiene
# una menor cantidad de registros. Obtenemos una precision de 76% y una exactitud del 77%. Este resultado es menos
# preciso que el dataset con outliers. Que el otro dataset sea mas preciso puede deberse a que esos outliers eran
# registros importantes o que el dataset sin outliers tiene muy pocos registros como para que el algoritmo sea bien
# entrenado.

print("\ngrafico logistica KNN para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_df_sin_outlier_grafico)

# En el grafico podemos ver una cantidad de falsos positivos y falsos negativos bastante alta, este modelo no es
# preciso. Tenemos un total de 63% de precision y 67% de exactitud. Este resultado es muy malo, el modelo no puede
# hacer buenas predicciones por lo que no es recomendable usarlo. En el grafico se puede observar mejor la inexactitud
# de este modelo. 

# Nuestro siguiente modelo se hara con maquinas de soporte vectorial con sus 4 tipos de kernel.

classifier_svc_linear = SVC(kernel="linear", random_state=0)
classifier_svc_poly = SVC(kernel="poly", random_state=0)
classifier_svc_rbf = SVC(kernel="rbf", random_state=0)
classifier_svc_sigmoid = SVC(kernel="sigmoid", random_state=0)

print("\nRegresion logistica classifier_svc_linear para dataset normal:")
logistic(df, classifier_svc_linear)

# Observamos una poca cantidad de falsos positivos y negativos, este modelo va a tener buena precision. Tenemos un total
# de 81% de precision y 84% de exactitud, este modelo puede dar buenas predicciones.

print("\nRegresion logistica classifier_svc_poly para dataset normal:")
logistic(df, classifier_svc_poly)

# Vemos que hay un poco mas de falsos positivos y negativos en este modelo que en el SVC lineal. Tenemos un total de 76%
# de precision y 80% de exactitud, este modelo es menos preciso que el SVC lineal.

print("\nRegresion logistica classifier_svc_rbf para dataset normal:")
logistic(df, classifier_svc_rbf)

# En este modelo vemos una cantidad similar de falsos positivos y falsos negativos que en el SVC lineal. Obtenemos
# un total de 79% de precision y 82% de exactitud, este modelo da resultados similares al SVC lineal. Es mejor que el
# SVC poly.

print("\nRegresion logistica classifier_svc_sigmoid para dataset normal:")
logistic(df, classifier_svc_sigmoid)

# En este modelo vemos tambien una cantidad similar de falsos positivos y falsos negativos que el SVC lineal. Obtenemos
# un total de 80% y una exactitud de 85%, es mas preciso que SVC rbf, pero un poquito menos que SVC lineal. Puede
# dar buenas predicciones.

# En conclusion, el SVC lineal es el mas preciso de todos con el dataset normal, el peor es el SVC poly.

print("\nGrafico Regresion logistica classifier_svc_linear para dataset normal:")
grafico_logistic(df, classifier_svc_linear)

# Se puede observar una gran cantidad de falsos negativos y positivos, el modelo no va a ser preciso. Tenemos un total
# de 63% de precision y un 69% de exactitud, este modelo no es capaz de dar buenas predicciones, no es recomendable
# usarlo.

print("\nGrafico Regresion logistica classifier_svc_poly para dataset normal:")
grafico_logistic(df, classifier_svc_poly)

# Por parte de este modelo vemos una cantidad de falsos negativos y falsos positivos mucho mayor al SVC lineal. Tenemos
# una precision de 56% y una exactitud de 64%, es menos preciso que el SVC lineal.

print("\nGrafico Regresion logistica classifier_svc_rbf para dataset normal:")
grafico_logistic(df, classifier_svc_rbf)

# En este modelo vemos una menor cantidad de falsos negativos y falsos positivos en comparacion al SVC poly. Hay una
# precision de 61% y una exactitud del 68%, es mas preciso que el SVC poly y tiene valores parecidos al de SVC lineal.

print("\nGrafico Regresion logistica classifier_svc_sigmoid para dataset normal:")
grafico_logistic(df, classifier_svc_sigmoid)

# En este modelo los falsos positivos y los falsos negativos aumentaron en comparacion con el SVC rbf. Hay un total de
# 55% de precision y un 59% de exactitud, es menos exacto que el SVC poly.

# En conclusion el SVC lieal es el mas preciso de todos en el dataset normal, el menos preciso es el SVC sigmoid.

# Continuaremos con el dataset sin outliers.

print("\nRegresion logistica classifier_svc_linear para dataset sin outlier:")
logistic(df_sin_outlier, classifier_svc_linear)

# Vemos poca cantidad de falsos positivos y negativos, el modelo va a ser bastante preciso. Tenemos un total de 75%
# de precision y una 79% de exactitud, este modelo podria dar buenas predicciones, pero no seria recomendable usarlo
# ya que su exactitud no esta dentro de lo aceptable.

print("\nRegresion logistica classifier_svc_poly para dataset sin outlier:")
logistic(df_sin_outlier, classifier_svc_poly)

# Podemos ver una mayor cantidad de falsos negativos y falsos positivos, el modelo no va a ser preciso. Tenemos 63%
# de precision y 68% de exactitud. El modelo es menos preciso que SVC lineal y no puede dar buenas predicciones.

print("\nRegresion logistica classifier_svc_rbf para dataset sin outlier:")
logistic(df_sin_outlier, classifier_svc_rbf)

# Vemos menos falsos positivos y negativos que en SVC poly. Obtenemos un 69% de precision y un 72% de exactitud, es
# mas preciso que SVC poly pero menos preciso que SVC lineal. No puede dar buenas predicciones.

print("\nRegresion logistica classifier_svc_sigmoid para dataset sin outlier:")
logistic(df_sin_outlier, classifier_svc_sigmoid)

# Hay un poco menos de falsos positivos y falsos negativos. Tenemos 71% de precision y 77% de exactitud, es mayor a
# SVC rbf pero menor a SVC lineal. Este metodo tampoco esta en lo aceptable para dar buenas predicciones.

# En conclusion, estos modelos con el dataset sin outliers dan menos precision que con el dataset normal y no es
# recomendable usar estos modelos.

print("\nGrafico Regresion logistica classifier_svc_linear para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_svc_linear)

# Hay una gran cantidad de falsos positivos y falsos negativos, el modelo no va a ser preciso. Nos da un 53% de
# precision y un 54% de exactitud, no puede dar buenas predicciones.

print("\nGrafico Regresion logistica classifier_svc_poly para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_svc_poly)

# Tenemos una menor cantidad de falsos negativos y positivos, igualmente el modelo no va a ser preciso. Tenemos un 59%
# de precision y un 65% de exactitud, es mejor que el SVC lienal pero no puede dar buenas predicciones.

print("\nGrafico Regresion logistica classifier_svc_rbf para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_svc_rbf)

# Volvemos a tener una gran cantidad de falsos positivos y falsos negativos. La precision es de 54% y la exactitud de
# 56%, es pareciso al SVC lieal y es peor que el SVC poly. No da buenas predicciones.

print("\nGrafico Regresion logistica classifier_svc_sigmoid para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_svc_sigmoid)

# Hay una inmensa cantidad de falsos positivos y falsos negativos. Tenemos un 45% de precision y un 44% de exactitud.
# este es el peor modelo de todos y no da buenas predicciones.

# En conclusion, el dataset sin outliers da unos peores resultados a comparacion con el dataset con los outliers.

# A continuacion utilizaremos el modelo de Naive Bayes para la clasificacion.

classifier = GaussianNB()

print("\nRegresion logistica Naive Bayes para dataset normal:")
logistic(df, classifier)

# En este modelo vemos una cantidad pequenia de falsos positivos y falsos negativos, el modelo va a ser preciso.
# Obtenemos un 81% de precision y un 82% de exactitud, el modelo puede dar buenas predicciones.

print("\nGrafico Regresion logistica Naive Bayes para dataset normal:")
grafico_logistic(df, classifier)

# En este modelo hay muchos falsos negativos y positivos, no va a ser preciso. Hay una precision de 61% de precision
# y un 68% de exactitud, este modelo no puede dar buenas predicciones, es es recomendable usarlo.

print("\nRegresion logistica Naive Bayes para dataset sin outlier:")
logistic(df_sin_outlier, classifier)

# Podemos ver que hay un poco mas de falsos positivos y negativos que en el dataset normal. Tenemos un 70% de precision
# y un 74% de exactitud, este modelo es menos preciso que el dataset normal, ademas no puede dar buenas predicciones.

print("\nGrafico Regresion logistica Naive Bayes para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier)

# Hay una gran cantidad de falsos negativos y positivos, el modelo no va a ser preciso. Tenemos un 54% de precision y
# un 56% de exactitud. Estos resultados son peores que con el dataset normal, no es recomendable usarlo.

# El proximo modelo lo haremos con arboles de decision para clasificacion.

classifier_tree = DecisionTreeClassifier(criterion="entropy", random_state=0)

print("\nRegresion logistica con arboles de decision para dataset normal:")
logistic(df, classifier_tree)

# Vemos una cantidad de falsos positivos y falsos negativos un poco elevada. Tenemos un total de 74% de precision y
# 77% de exactitud, este modelo no es muy preciso. Esto puede deberse a que el arbol de decision necesita una mayor
# cantidad de datos para poder ser mas exacto.

print("\nGrafico Regresion logistica con arboles de decision para dataset normal:")
grafico_logistic(df, classifier_tree)

# En este modelo hay una gran cantidad de falsos negativos y falsos positivos, no es preciso. Tenemos 52% en precision
# y 55% de exactitud, no puede dar buenas predicciones.

print("\nRegresion logistica con arboles de decision para dataset sin outlier:")
logistic(df_sin_outlier, classifier_tree)

# Hay una cantida de falsos positivos y negativos elevada, este modelo no va a ser preciso. Tenemos un total de 70% de
# precision y un 74% de exactitud, es menos preciso que el modelo con el dataset normal, debido a que hay una menor
# cantidad de datos en este dataset.

print("\nGrafico Regresion logistica con arboles de decision para dataset sin outlier:")
grafico_logistic(df_sin_outlier, classifier_tree)


# Aqui vemos una cantidad de falsos negativos y positivos elevado. Tenemos un 57% de precision y un 56% de exactitud.
# Este modelo es mejor que con el dataset normal, aun asi no puede dar buenas predicciones y es recomendable no usarlo.

# El ultimo modelo que utilizaremos para poder hacer la clasificacion sera random forest. Primero necesitamos definir
# una funcion que nos diga cual es la mejor profundidad del arbol, para asi sacar el mejor resultado del modelo.

def max_depth_random_forest(df):
    X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Valores de max_depth a evaluar.
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Lista para almacenar los resultados de la validación cruzada.
    scores = []

    # Iterar sobre los valores de max_depth.
    for max_depth in max_depth_values:
        rf_classifier = RandomForestClassifier(max_depth=max_depth)

        cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

        scores.append(np.mean(cv_scores))

    # Encontrar el índice del mejor valor de max_depth.
    best_index = np.argmax(scores)
    best_max_depth = max_depth_values[best_index]

    print("Resultados de la validación cruzada:")
    print("Mejor max_depth: {}".format(best_max_depth))

    # Una vez encontrado la mejor profundidad procedemos a realizar la clasificacion.

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, max_depth=best_max_depth)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Predicción', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Matriz de Confusión', fontsize=17)
    plt.show()

    print(f'\nPrecisión del modelo: {precision_score(y_test, y_pred):.2f}')
    print(f'Exactitud del modelo:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Sensibilidad del modelo: {recall_score(y_test, y_pred):.2f}')
    print(f'Puntaje F1 del modelo:{f1_score(y_test, y_pred):.2f}')
    print(f'Curva ROC - AUC del modelo:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))


def grafico_max_depth_random_forest(df):
    X = df[["age", "oldpeak"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Valores de max_depth a evaluar.
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Lista para almacenar los resultados de la validación cruzada.
    scores = []

    # Iterar sobre los valores de max_depth.
    for max_depth in max_depth_values:
        rf_classifier = RandomForestClassifier(max_depth=max_depth)

        cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

        scores.append(np.mean(cv_scores))

    # Encontrar el índice del mejor valor de max_depth.
    best_index = np.argmax(scores)
    best_max_depth = max_depth_values[best_index]

    print("Resultados de la validación cruzada:")
    print("Mejor max_depth: {}".format(best_max_depth))

    # Una vez encontrado la mejor profundidad procedemos a realizar la clasificacion.

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, max_depth=best_max_depth)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Predicción', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Matriz de Confusión', fontsize=17)
    plt.show()

    print(f'\nPrecisión del modelo: {precision_score(y_test, y_pred):.2f}')
    print(f'Exactitud del modelo:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Sensibilidad del modelo: {recall_score(y_test, y_pred):.2f}')
    print(f'Puntaje F1 del modelo:{f1_score(y_test, y_pred):.2f}')
    print(f'Curva ROC - AUC del modelo:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Clasificador (Conjunto de Entrenamiento)')
    plt.xlabel('Edad')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Clasificador (Conjunto de Test)')
    plt.xlabel('Edad')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()


print("\nRandom forest para dataset con outliers:")
max_depth_random_forest(df)

# Vemos que la mejor profundidad para este dataset es de 18, en el grafico de falsos positivos y falsos negativos vemos
# una poca cantidad de estos casos. Finalmente tenemos una precision de 85% y una exactitud de 85%, este modelo es muy
# bueno y puede dar buenas predicciones.

print("\nRandom forest para dataset sin outliers:")
max_depth_random_forest(df_sin_outlier)

# Por otro lado podemos ver que en el dataset sin outliers la mejor profundidad es de 2, esto quiere decir que tenemos
# muy pocos registros, o estos mismos no son muy buenos con random forest, el hecho de que no podamos tener una mejor
# precision a una mayor profundidad nos deja ver que este modelo no es bueno. Tambien vemos una gran cantidad de falsos
# positivos y negativos, este modelo no es preciso. Nos da un 62% de precision y 67% de exactitud. El modelo no puede
# dar buenas predicciones.

print("\nGrafico Random forest para dataset con outliers:")
grafico_max_depth_random_forest(df)

# Vemos que para este modelo la mejor profundidad es de 1, osea que este modelo es muy malo, se puede ver tambien con
# la gran cantidad de falsos positivos y negativos. Tenemos un 61% de precision y un 68% de exactitud. Este modelo no
# puede dar buenas predicciones.

print("\nGrafico Random forest para dataset sin outliers:")
grafico_max_depth_random_forest(df_sin_outlier)

# En este modelo la mejor profundidad es de 3, un poco mas que el anterior. Tambien hay gran cantidad de falsos
# positivos y negativos. Sin embargo hay una precision de 54% y una exactitud de 56%. Este modelo es muy malo para hacer
# predicciones.

# Ahora que realizamos todas las clasificaciones posibles vamos a sacar las conclusiones finales:

# Observamos que en todos los tipos de clasificacion, el dataset con outliers tuvo una mayor precision que el dataset
# sin outliers. Esto puede deberse a la poca cantidad de datos del dataset o que los outliers en realidad eran datos
# bastante importantes para el modelo.

# De todos los modelos, el de los mejores resultados es con el Random Forest con profundidad de 18 con un 85% de
# exactitud y un 85% de precision. Tambien KNN con 9 vecinos nos dio buenos resultados con 82% de precision y 88% de
# exactitud. El mayor problema de KNN es que debemos ingresar la cantidad vecinos (k) adecuada para tener buenos
# resultados y el random forest con una mayor cantidad de registros seria mas preciso.

# Tambien podemos ver que las variables que utilizamos para realizar los graficos no son muy compatibles, ya que nos
# dio en todos los casos con ambos dataset muy malos resultados. Al final la mejor opcion es utilizar todas las
# variables que tengan buena correlacion con la variable que queremos predecir.
