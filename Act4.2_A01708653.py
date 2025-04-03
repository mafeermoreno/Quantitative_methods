# María Fernanda Moreno Gómez

# Actividad 4.2. Similitud en textos mediante BoW

'''
Instrucciones


1. Implementa la vectorización de textos mediante Bag of Words, con la medida de distancia del coseno para obtener un valor de similitud.

2. Utiliza el dataset de Quora questions para determinar la similitud. Crea una nueva columna llamada cosine_distance donde se registre el resultado. Además, agrega dos columnas más llamadas q1_vector y q2_vector, donde colocarás la codificación de cada pregunta.

Nota: observa que la medida de duplicated determina si la pregunta es la misma o no. En nuestro caso, solamente estaremos midiendo qué tan similares son.

Entregables: código fuente y archivo CSV generado.
'''

# Librerías
import pandas as pd
import numpy as np
import re
from collections import Counter

# Leer el dataset
df = pd.read_csv('questions.csv', encoding='ISO-8859-1')

# Preprocesar los datos
def preprocess_text(text):
    if pd.isna(text) or text.strip() == "":  # Verificar si el texto es NaN o está vacío
        return "empty"
    text = str(text).lower()  # Convertir a minúsculas
    text = re.sub(r'[\"()\[\]{}:?,.!\-]', '', text)  # Quitar signos de puntuación
    text = re.sub(r'\s+', ' ', text).strip()  # Reemplazar espacios dobles por uno solo
    return text

# Aplicar preprocesamiento a las preguntas
df['question1'] = df['question1'].apply(preprocess_text)
df['question2'] = df['question2'].apply(preprocess_text)

# Función para convertir texto en vector usando Bag of Words
def text_to_vector(text):
    words = text.split()
    return Counter(words)

# Función para calcular la similitud del coseno
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[word] * vec2[word] for word in intersection)
    sum1 = sum(val**2 for val in vec1.values())
    sum2 = sum(val**2 for val in vec2.values())
    denominator = np.sqrt(sum1) * np.sqrt(sum2)
    if not denominator:
        return 0.0
    return numerator / denominator

# Aplicar vectorización y calcular similitud del coseno
df['vector_q1'] = df['question1'].apply(text_to_vector)
df['vector_q2'] = df['question2'].apply(text_to_vector)
df['coseno'] = df.apply(lambda row: cosine_similarity(row['vector_q1'], row['vector_q2']), axis=1)

# Guardar resultado en CSV
df.to_csv('questions_with_similarity.csv', index=False)

print(df.head(10))