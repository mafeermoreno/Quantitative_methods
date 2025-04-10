# María Fernanda Moreno Gómez

# Actividad 4.3. Similitud en textos mediante TF-IDF y Cadenas de Markov

'''
Instrucciones


1. Agrega a tu código de similitud por BoW la vectorización de textos mediante TF-IDF y Cadenas de Markov, con la medida de distancia del coseno para obtener un valor de similitud.

2. Utiliza el dataset de Quora questions para determinar la similitud. Crea columnas llamadas cos_BOW, cos_TFID y cos_MARK donde se registren los resultados. Para los vectores de BoW y TF-IDF, llámalos en el excel como q1_vecBoW, q2_vecBoW, q1_vecTFIDF, etc. Para las matrices de las cadenas de Markov, utiliza algún método (como flatten()) para representarlas como vector y agrégalos como q1_vecMark, etc. 

Nota: observa que la medida de duplicated determina si la pregunta es la misma o no (0 o 1). En nuestro caso, solamente estaremos midiendo qué tan similares son.

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

# Actividad 4.3. Similitud en textos mediante TF-IDF y Cadenas de Markov

# Selecciona una fila para analizar (por ejemplo, la primera)
fila = df.iloc[0]
q1 = fila["question1"]
q2 = fila["question2"]

# Obtener el vocabulario de ambas preguntas
def get_vocab(*texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return sorted(vocab)

# Calcular TF
def compute_tf(text, vocab):
    words = text.split()
    total_words = len(words)
    return {word: words.count(word) / total_words for word in vocab}

# Calcular IDF
def compute_idf(vocab, texts):
    N = len(texts)
    return {
        word: np.log(N / (sum(word in t.split() for t in texts) + 1)) + 1
        for word in vocab
    }

# Calcular TF-IDF
def compute_tfidf(tf, idf):
    return {word: tf[word] * idf[word] for word in tf}

# Procesar
vocabulario = get_vocab(q1, q2)
tf_q1 = compute_tf(q1, vocabulario)
tf_q2 = compute_tf(q2, vocabulario)
idf = compute_idf(vocabulario, [q1, q2])
tfidf_q1 = compute_tfidf(tf_q1, idf)
tfidf_q2 = compute_tfidf(tf_q2, idf)

# Crear la tabla
tabla = pd.DataFrame({
    "Palabra": vocabulario,
    "TF(q1)": [tf_q1[w] for w in vocabulario],
    "TF(q2)": [tf_q2[w] for w in vocabulario],
    "IDF": [idf[w] for w in vocabulario],
    "TF-IDF(q1)": [tfidf_q1[w] for w in vocabulario],
    "TF-IDF(q2)": [tfidf_q2[w] for w in vocabulario],
})

# Mostrar la tabla completa o parcial
print(tabla.head(15)) 