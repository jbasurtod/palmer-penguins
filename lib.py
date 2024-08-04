import os # Proporciona funciones para interactuar con el sistema operativo.
import pandas as pd # Manipulación y análisis de datos tabulares (filas y columnas).
import numpy as np # Operaciones numéricas y matriciales.
import seaborn as sns # Visualización estadística de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA # Implementación del Análisis de Componentes Principales (PCA).
from sklearn.preprocessing import StandardScaler # Estandarización de datos para análisis estadísticos.
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

def GetStats(df0,df1):
    df0 = rename_columns(df0, "__0")
    df1 = rename_columns(df1, "__1")
    variables0 = list(df0.columns)
    variables1 = list(df1.columns)

    # Calcula las estadísticas descriptivas para cada variable y crea un DataFrame con los resultados.
    return pd.concat([
        pd.DataFrame({
        'Mínimo': df0[variables0].min(),
        'Percentil 25': df0[variables0].quantile(0.25),
        'Mediana': df0[variables0].median(),
        'Percentil 75': df0[variables0].quantile(0.75),
        'Media': df0[variables0].mean(),
        'Máximo': df0[variables0].max(),
        'Desviación Estándar': df0[variables0].std(),
        'Varianza': df0[variables0].var(),
        'Datos Perdidos': df0[variables0].isna().sum()  # Cuenta los valores NaN por variable.
    }).round(2),
        pd.DataFrame({
        'Mínimo': df1[variables1].min(),
        'Percentil 25': df1[variables1].quantile(0.25),
        'Mediana': df1[variables1].median(),
        'Percentil 75': df1[variables1].quantile(0.75),
        'Media': df1[variables1].mean(),
        'Máximo': df1[variables1].max(),
        'Desviación Estándar': df1[variables1].std(),
        'Varianza': df1[variables1].var(),
        'Datos Perdidos': df1[variables1].isna().sum()  # Cuenta los valores NaN por variable.
    }).round(2)],axis=0).sort_index()

def plot_pca_scatter_with_vectors(pca, datos_estandarizados, n_components, components_):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados
    con vectores de las correlaciones escaladas entre variables y componentes

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
        components_: Array con las componentes.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones y variables en PCA')
            
            
            # Añadimos vectores que representen las correlaciones escaladas entre variables y componentes
            fit = pca.fit(datos_estandarizados)
            coeff = np.transpose(fit.components_)
            scaled_coeff = 8 * coeff  #8 = escalado utilizado, ajustar en función del ejemplo
            for var_idx in range(scaled_coeff.shape[0]):
                plt.arrow(0, 0, scaled_coeff[var_idx, i], scaled_coeff[var_idx, j], color='red', alpha=0.5)
                plt.text(scaled_coeff[var_idx, i], scaled_coeff[var_idx, j],
                     datos_estandarizados.columns[var_idx], color='red', ha='center', va='center')
            
            plt.show()
def rename_columns(df, custom_string):
    # Create a dictionary with the new column names
    new_column_names = {col: f"{col}_{custom_string}" for col in df.columns}
    
    # Rename the columns using the dictionary
    df.rename(columns=new_column_names, inplace=True)
    
    return df.copy()
def SilhouetteGraph(clusters,df_std):
    # Configuramos el modelo KMeans con n clusters y un estado aleatorio fijo
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    # Ajustamos el modelo KMeans a los datos estandarizados
    kmeans.fit(df_std)
    # Obtenemos las etiquetas de clusters resultantes
    labels = kmeans.labels_
        
    # Calculamos los valores de silueta para cada observación
    silhouette_values = silhouette_samples(df_std, labels)
        
    # Configuramos el tamaño de la figura para el gráfico
    plt.figure(figsize=(8, 6))
    y_lower = 10  # Inicio del margen inferior en el gráfico
    
    # Iteramos sobre los 4 clusters para calcular los valores de silueta y dibujar el gráfico
    for i in range(4):
        # Extraemos los valores de silueta para las observaciones en el cluster i
        ith_cluster_silhouette_values = silhouette_values[labels == i]
        # Ordenamos los valores para que el gráfico sea más claro
        ith_cluster_silhouette_values.sort()
        
        # Calculamos donde terminarán las barras de silueta en el eje y
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Elegimos un color para el cluster
        color = plt.cm.get_cmap("inferno")(float(i) / 4)
        # Rellenamos el gráfico entre un rango en el eje y con los valores de silueta
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Etiquetamos las barras de silueta con el número de cluster en el eje y
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Actualizamos el margen inferior para el siguiente cluster
        y_lower = y_upper + 10  # 10 para el espacio entre clusters
    
    # Títulos y etiquetas para el gráfico
    plt.title("Gráfico de Silueta para los Clusters")
    plt.xlabel("Valores del Coeficiente de Silueta")
    plt.ylabel("Etiqueta del Cluster")
    plt.grid(True)  # Añadimos una cuadrícula para mejor legibilidad
    plt.show()  # Mostramos el gráfico resultante

def GraficarClusters(num_clusters,linkage_matrix,penguins,df_std):
    #num_clusters =2
    cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    #print("Cluster Assignments:", cluster_assignments) 
    print("Gráfico para "+str(num_clusters)+" clusters")
    # Creamos una nueva columna 'Cluster4' y asignamos los valores de 'cluster_assignments' a ella
    # Ahora 'df' contiene una nueva columna 'Cluster4' con las asignaciones de cluster
    penguins['Cluster4'] = cluster_assignments
    
    # Visualización de la distribución espacial de los clusters
    # Paso 1: Realizar PCA
    pca = PCA(n_components=2)  # Inicializamos PCA para 2 componentes principales
    eliminar = ['Cluster4']
    principal_components = pca.fit_transform(penguins.drop(eliminar, axis=1))  # Transformamos los datos a 2 componentes
    
    
    fit = pca.fit(df_std)
    
    # Obtener los autovalores asociados a cada componente principal.
    autovalores = fit.explained_variance_
    
    variables = list(df_std.columns) 
    # Obtener los autovectores asociados a cada componente principal y transponerlos.
    autovectores = pd.DataFrame(pca.components_.T, 
                                columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                                index = ['{}_z'.format(variable) for variable in variables])
    
    # Calculamos las dos primeras componentes principales
    resultados_pca = pd.DataFrame(fit.transform(df_std), 
                                  columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                                  index=df_std.index)
    
    # Añadimos las componentes principales a la base de datos estandarizada.
    df_z_cp = pd.concat([df_std, resultados_pca], axis=1)
    
    # Calculo la matriz de correlaciones entre veriables y componentes
    Correlaciones_var_comp = df_z_cp.corr()
    Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]
    
    # Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
    var_explicada = fit.explained_variance_ratio_*100
    
    # Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
    var_acumulada = np.cumsum(var_explicada)
    
    # Crear un DataFrame de pandas con los datos anteriores y establecer índice.
    data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
    tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)]) 
    
    # Imprimir la tabla
    print(tabla)
    
    # Creamos un nuevo DataFrame para los componentes principales 2D
    # Nos aseguramos de que df_pca tenga el mismo índice que df
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=penguins.index)
    
    # Paso 2: Crear un gráfico de dispersión con colores para los clusters
    plt.figure(figsize=(10, 8))  # Establecemos el tamaño del gráfico
    
    # Recorremos las asignaciones únicas de clusters y trazamos puntos de datos con el mismo color
    for cluster in np.unique(cluster_assignments):
        cluster_indices = df_pca.loc[cluster_assignments == cluster].index
        plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                    df_pca.loc[cluster_indices, 'PC2'],
                    label=f'Cluster {cluster}')  # Etiqueta para cada cluster
        # Anotamos cada punto con el nombre del país
        for i in cluster_indices:
            plt.annotate(i,
                         (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']), fontsize=10,
                         textcoords="offset points",  # cómo posicionar el texto
                         xytext=(0,10),  # distancia del texto a los puntos (x,y)
                         ha='center')  # alineación horizontal puede ser izquierda, derecha o centro
    
    plt.title("Gráfico de PCA 2D con Asignaciones de Cluster")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid()
    plt.show()

def plot_cos2_heatmap(cosenos2):
    """
    Genera un mapa de calor (heatmap) de los cuadrados de las cargas en las Componentes Principales (cosenos al cuadrado).

    Args:
        cosenos2 (pd.DataFrame): DataFrame de los cosenos al cuadrado, donde las filas representan las variables y las columnas las Componentes Principales.

    """
    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar 'cos2' con un solo color
    sns.heatmap(cosenos2, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Cuadrados de las Cargas en las Componentes Principales')

    # Muestra el gráfico
    plt.show()

def plot_cos2_bars(cos2):
    """
    Genera un gráfico de barras para representar la varianza explicada de cada variable utilizando los cuadrados de las cargas (cos^2).

    Args:
        cos2 (pd.DataFrame): DataFrame que contiene los cuadrados de las cargas de las variables en las componentes principales.

    Returns:
        None
    """
    # Crea una figura de tamaño 8x6 pulgadas para el gráfico
    plt.figure(figsize=(8, 6))

    # Crea un gráfico de barras para representar la varianza explicada por cada variable
    sns.barplot(x=cos2.sum(axis=1), y=cos2.index, color="blue")

    # Etiqueta los ejes
    plt.xlabel('Suma de los $cos^2$')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Varianza Explicada de cada Variable por las Componentes Principales')

    # Muestra el gráfico
    plt.show()

def plot_contribuciones_proporcionales(cos2, autovalores, n_components):
    """
    Cacula las contribuciones de cada variable a las componentes principales y
    Genera un gráfico de mapa de calor con los datos
    Args:
        cos2 (DataFrame): DataFrame de los cuadrados de las cargas (cos^2).
        autovalores (array): Array de los autovalores asociados a las componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Calcula las contribuciones multiplicando cos2 por la raíz cuadrada de los autovalores
    contribuciones = cos2 * np.sqrt(autovalores)

    # Inicializa una lista para las sumas de contribuciones
    sumas_contribuciones = []

    # Calcula la suma de las contribuciones para cada componente principal
    for i in range(n_components):
        nombre_componente = f'Componente {i + 1}'
        suma_contribucion = np.sum(contribuciones[nombre_componente])
        sumas_contribuciones.append(suma_contribucion)

    # Calcula las contribuciones proporcionales dividiendo por las sumas de contribuciones
    contribuciones_proporcionales = contribuciones.div(sumas_contribuciones, axis=1) * 100

    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar las contribuciones proporcionales
    sns.heatmap(contribuciones_proporcionales, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Contribuciones Proporcionales de las Variables en las Componentes Principales')

    # Muestra el gráfico
    plt.show()
    
    # Devuelve los DataFrames de contribuciones y contribuciones proporcionales
    return contribuciones_proporcionales

def plot_varianza_explicada(var_explicada, n_components):
    """
    Representa la variabilidad explicada por cada componente principal
    Args:
      var_explicada (array): Un array que contiene el porcentaje de varianza explicada
        por cada componente principal. Generalmente calculado como
        var_explicada = fit.explained_variance_ratio_ * 100.
      n_components (int): El número total de componentes principales.
        Generalmente calculado como fit.n_components.
    """  
    # Crear un rango de números de componentes principales de 1 a n_components
    num_componentes_range = np.arange(1, n_components + 1)

    # Crear una figura de tamaño 8x6
    plt.figure(figsize=(8, 6))

    # Trazar la varianza explicada en función del número de componentes principales
    plt.plot(num_componentes_range, var_explicada, marker='o')

    # Etiquetas de los ejes x e y
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')

    # Título del gráfico
    plt.title('Variabilidad Explicada por Componente Principal')

    # Establecer las marcas en el eje x para que coincidan con el número de componentes
    plt.xticks(num_componentes_range)

    # Mostrar una cuadrícula en el gráfico
    plt.grid(True)

    # Agregar barras debajo de cada punto para representar el porcentaje de variabilidad explicada
    # - 'width': Ancho de las barras de la barra. En este caso, se establece en 0.2 unidades.
    # - 'align': Alineación de las barras con respecto a los puntos en el eje x. 
    #   'center' significa que las barras estarán centradas debajo de los puntos.
    # - 'alpha': Transparencia de las barras. Un valor de 0.7 significa que las barras son 70% transparentes.
    plt.bar(num_componentes_range, var_explicada, width=0.2, align='center', alpha=0.7)

    # Mostrar el gráfico
    plt.show()

def plot_corr_cos(n_components, correlaciones_datos_con_cp):
    """
    Genera un gráfico en el que se representa un vector por cada variable, usando como ejes las componentes, la orientación
    y la longitud del vector representa la correlación entre cada variable y dos de las componentes. El color representa el
    valor de la suma de los cosenos al cuadrado.
    
    Args:
        n_components (int): Número entero que representa el número de componentes principales seleccionadas.
        correlaciones_datos_con_cp (DataFrame): DataFrame que contiene la matriz de correlaciones entre variables y componentes
    """
    # Definir un mapa de color (cmap) sensible a las diferencias numéricas
    cmap = plt.get_cmap('coolwarm')  # Puedes ajustar el cmap según tus preferencias
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los cosenos al cuadrado
            sum_cos2 = correlaciones_datos_con_cp.iloc[:, i] ** 2 + correlaciones_datos_con_cp.iloc[:, j] ** 2
            
            # Crear un nuevo gráfico para cada par de componentes principales
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Dibujar un círculo de radio 1
            circle = plt.Circle((0, 0), 1, fill=False, color='b', linestyle='dotted')
            ax.add_patch(circle)
            
            # Dibujar vectores para cada variable con colores basados en la suma de los cosenos al cuadrado
            for k, var_name in enumerate(correlaciones_datos_con_cp.index):
                x = correlaciones_datos_con_cp.iloc[k, i]  # Correlación en la primera dimensión
                y = correlaciones_datos_con_cp.iloc[k, j]  # Correlación en la segunda dimensión
                
                # Seleccionar un color de acuerdo a la suma de los cosenos al cuadrado
                color = cmap(sum_cos2.iloc[k])
                
                # Dibujar el vector con el color seleccionado
                ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)
                
                # Agregar el nombre de la variable junto a la flecha con el mismo color
                ax.text(x, y, var_name, color=color, fontsize=12, ha='right', va='bottom')
            
            # Dibujar líneas discontinuas que representen los ejes
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            ax.set_xlabel(f'Componente Principal {i + 1}')
            ax.set_ylabel(f'Componente Principal {j + 1}')
            
            # Establecer los límites del gráfico
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            
            # Agregar un mapa de color (colorbar) y su leyenda
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])  # Evita errores de escala
            plt.colorbar(sm, ax=ax, orientation='vertical', label='cos^2')  # Agrega la leyenda
            
            # Mostrar el gráfico
            plt.grid()
            plt.show()

def plot_pca_scatter(pca, datos_estandarizados, n_components):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados.

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')
            
            plt.show()
            
def plot_pca_scatter_with_categories(datos_componentes_sup_var, componentes_principales_sup, n_components, var_categ):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados con categorías.

    Args:
        datos_componentes_sup_var (pd.DataFrame): DataFrame que contiene las categorías.
        componentes_principales_sup (np.ndarray): Matriz de componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
        var_categ (str): Nombre de la variable introducida
    """
    # Obtener las categorías únicas
    categorias = datos_componentes_sup_var[var_categ].unique()

    # Iterar sobre todos los posibles pares de componentes principales
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Crear un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))
            plt.scatter(componentes_principales_sup[:, i], componentes_principales_sup[:, j])

            for categoria in categorias:
                # Filtrar las observaciones por categoría
                observaciones_categoria = componentes_principales_sup[datos_componentes_sup_var[var_categ] == categoria]
                # Calcular el centroide de la categoría
                centroide = np.mean(observaciones_categoria, axis=0)
                plt.scatter(centroide[i], centroide[j], label=categoria, s=100, marker='o')

            # Añadir etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_componentes_sup_var.index)

            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales_sup[k, i], componentes_principales_sup[k, j]))

            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')

            # Establecer el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')

            # Mostrar la leyenda para las categorías
            plt.legend()
            plt.show()