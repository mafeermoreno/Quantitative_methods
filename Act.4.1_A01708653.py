#María Fernanda Moreno Gómez
# Actividad 4.1: Tablas de frecuencias e histogramas.

'''
Instrucciones:
Utilizando Python, de manera individual, escribe un programa que construya la tabla de frecuencias de un conjunto de datos. Además, tu programa deberá graficar el histograma.
'''

# Librerías
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from tabulate import tabulate

# Función para cargar los datos
def cargar_datos(filename):
    return np.loadtxt(filename)

# Función para redondear hacia arriba con una cantidad de decimales específica
def redondeo_techo(valor, decimales):
    factor = 10 ** decimales
    return math.ceil(valor * factor) / factor

# Función para redondear hacia abajo con una cantidad de decimales específica
def redondeo_piso(valor, decimales):
    factor = 10 ** decimales
    return math.floor(valor * factor) / factor

# Función para calcular los parámetros de la tabla de frecuencias
def obtener_parametros(data, decimales):
    N = len(data) 
    min_val = np.min(data)
    max_val = np.max(data)

    # Redondear los valores
    min_val = redondeo_piso(min_val, decimales)  # Redondear hacia abajo
    max_val = redondeo_techo(max_val, decimales)  # Redondear hacia arriba
    C = int(math.ceil(1 + 3.30 * np.log10(N)))  # Número de clases
    W = redondeo_techo((max_val - min_val) / C, decimales)  # Ancho de intervalo redondeado hacia arriba

    return N, min_val, max_val, C, W

# Función para calcular la tabla de frecuencias
def calcular_tabla_frecuencias(data, decimales):
    N, min_val, max_val, C, W = obtener_parametros(data, decimales)

    # Crear los límites de los intervalos con redondeo adecuado
    bins = [round(min_val + i * W, decimales) for i in range(C + 1)]
    frecuencias, _ = np.histogram(data, bins=bins)

    # Crear los intervalos como texto para la tabla
    intervalos = [f"[{bins[i]:.{decimales}f} - {bins[i+1]:.{decimales}f})" for i in range(len(bins) - 1)]
    
    tabla = pd.DataFrame({"Intervals": intervalos, "f": frecuencias})
    return tabla, bins[:-1], frecuencias  

# Función para graficar el histograma
def graficar_histograma(bins, frecuencias, title):
    # Espaciado entre barras
    width = 0.8 * (bins[1] - bins[0])

    plt.bar(bins, frecuencias, width=width, color='pink', edgecolor='darkred', align='edge')

    # Agregar los intervalos como etiquetas en las barras
    for i in range(len(frecuencias)):
        if i < len(bins) - 1:
            plt.text(bins[i] + width / 2, frecuencias[i] + 2, f'[{bins[i]:.2f} - {bins[i+1]:.2f})', 
                     ha='center', color='black', fontsize=9)

    plt.xlabel('Intervalo')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Función principal
def main():
    archivos = ['data01.txt', 'data02.txt', 'data03.txt']
    decimales_dict = {
        'data01.txt': 4,
        'data02.txt': 2,
        'data03.txt': 3
    }

    for archivo in archivos:
        decimales = decimales_dict[archivo]
        print(f"\nTabla de frecuencias para {archivo}:")
        data = cargar_datos(archivo)
        tabla, bins, frecuencias = calcular_tabla_frecuencias(data, decimales)
        print(tabulate(tabla, headers='keys', tablefmt='pretty'))
        print(f"\nSum of frequencies: {sum(frecuencias)}")
        graficar_histograma(bins, frecuencias, f'Histograma de {archivo}')

if __name__ == "__main__":
    main()