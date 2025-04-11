# María Fernanda Moreno Gómez
# Actividad 4.3. Similitud en textos mediante TF-IDF y Cadenas de Markov

import pandas as pd
import numpy as np
import re
from collections import Counter

# Leer solo las primeras 10 filas del dataset
try:
    df = pd.read_csv('questions.csv', encoding='ISO-8859-1', nrows=10)
    print(f"Se cargaron {len(df)} filas del dataset")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    df = None

if df is not None:
    # Preprocesamiento de texto
    def preprocess_data(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text) 
        return text.split()  

    df["tokens_q1"] = df["question1"].apply(preprocess_data)
    df["tokens_q2"] = df["question2"].apply(preprocess_data)

    # Crear una tabla que contenga todas las preguntas únicas
    unique_questions = pd.concat([
        df[["qid1", "tokens_q1"]].rename(columns={"qid1": "qid", "tokens_q1": "tokens"}),
        df[["qid2", "tokens_q2"]].rename(columns={"qid2": "qid", "tokens_q2": "tokens"})
    ])
    unique_questions = unique_questions.drop_duplicates("qid").set_index("qid")
    all_documents = unique_questions["tokens"].tolist()
    
    # Calcular IDF (Inverse Document Frequency)
    def idf_calculation(documents):
        N = len(documents)
        unique_words = set(word for doc in documents for word in doc)
        idf_values = {}
        
        for word in unique_words:
            df_count = sum(1 for doc in documents if word in doc)
            idf_values[word] = np.log(N / (df_count + 1)) + 1
        
        return idf_values
    
    idf = idf_calculation(all_documents)
    
    # Calcular TF (Term Frequency) para un documento
    def tf_calculation(document):
        word_count = len(document)
        if word_count == 0:
            return {}
        
        counter = Counter(document)
        return {word: count / word_count for word, count in counter.items()}
    
    # Función de similitud Bag of Words con coseno
    def cosine_bow(q1, q2):
        vocab = sorted(list(set(q1) | set(q2)))
        counter_q1 = Counter(q1)
        counter_q2 = Counter(q2)
        
        vec1 = [counter_q1.get(word, 0) for word in vocab]
        vec2 = [counter_q2.get(word, 0) for word in vocab]
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot / (norm1 * norm2) if norm1 and norm2 else 0, vec1, vec2
    
    # Función de similitud TF-IDF con coseno
    def cosine_tfidf(q1, q2, idf_dict):
        tf_q1 = tf_calculation(q1)
        tf_q2 = tf_calculation(q2)
        
        vocab = sorted(list(set(q1) | set(q2)))
        
        tfidf1 = [tf_q1.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
        tfidf2 = [tf_q2.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
        
        dot = np.dot(tfidf1, tfidf2)
        norm1 = np.linalg.norm(tfidf1)
        norm2 = np.linalg.norm(tfidf2)
        
        return dot / (norm1 * norm2) if norm1 and norm2 else 0, tfidf1, tfidf2
    
    # Función de similitud con cadenas de Markov
    def markov_similarity(q1, q2):
        # Si alguna de las preguntas es demasiado corta, no podemos crear una matriz de transición
        if len(q1) <= 1 or len(q2) <= 1:
            return 0, [], []
        
        all_tokens = sorted(list(set(q1 + q2)))
        token_index = {token: i for i, token in enumerate(all_tokens)}
        n = len(all_tokens)
        
        # Crear matrices de transición
        mat1 = np.zeros((n, n))
        mat2 = np.zeros((n, n))
        
        # Llenar matriz para q1
        for i in range(len(q1)-1):
            current = token_index[q1[i]]
            next_token = token_index[q1[i+1]]
            mat1[current][next_token] += 1
        
        # Normalizar filas de mat1
        row_sums = mat1.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Evitar división por cero
        mat1 = mat1 / row_sums[:, np.newaxis]
        
        # Llenar matriz para q2
        for i in range(len(q2)-1):
            current = token_index[q2[i]]
            next_token = token_index[q2[i+1]]
            mat2[current][next_token] += 1
        
        # Normalizar filas de mat2
        row_sums = mat2.sum(axis=1)
        row_sums[row_sums == 0] = 1
        mat2 = mat2 / row_sums[:, np.newaxis]
        
        # Aplanar matrices y calcular similitud de coseno
        flat1 = mat1.flatten()
        flat2 = mat2.flatten()
        
        dot = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        return dot / (norm1 * norm2) if norm1 and norm2 else 0, flat1.tolist(), flat2.tolist()
    
    # Calcular similitudes para cada par de preguntas
    bow_results = df.apply(lambda row: cosine_bow(row["tokens_q1"], row["tokens_q2"]), axis=1)
    df["bag_of_words_cosine"] = bow_results.apply(lambda x: x[0])
    df["vector_q1"] = bow_results.apply(lambda x: x[1])
    df["vector_q2"] = bow_results.apply(lambda x: x[2])
    
    tfidf_results = df.apply(lambda row: cosine_tfidf(row["tokens_q1"], row["tokens_q2"], idf), axis=1)
    df["IDF_cosine"] = tfidf_results.apply(lambda x: x[0])
    df["TF(text1)*IDF"] = tfidf_results.apply(lambda x: x[1])
    df["TF(text2)*IDF"] = tfidf_results.apply(lambda x: x[2])
    
    markov_results = df.apply(lambda row: markov_similarity(row["tokens_q1"], row["tokens_q2"]), axis=1)
    df["markov_cosine"] = markov_results.apply(lambda x: x[0])
    df["markov_matrix_q1"] = markov_results.apply(lambda x: x[1])
    df["markov_matrix_q2"] = markov_results.apply(lambda x: x[2])
    
    # Crear una copia final con todas las columnas requeridas (tanto las tuyas como las de tu compañero)
    df_final = df[[
        "id", "qid1", "qid2", "question1", "question2", "is_duplicate",
        "vector_q1", "vector_q2", "bag_of_words_cosine",
        "TF(text1)*IDF", "TF(text2)*IDF", "IDF_cosine",
        "markov_matrix_q1", "markov_matrix_q2", "markov_cosine"
    ]]
    
    # Guardar resultados en CSV
    output_path = "questions_with_all_similarities.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"\nCSV creado: {output_path}")