import pandas as pd
import numpy as np
import re
import os
from collections import Counter

# Preprocesar los datos para poner el texto en minúsculas, eliminar puntuación y tokenizar
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  
    return text.split()  

# Calcular el idf para todos los términos de los documentos
def compute_idf(documents):
    N = len(documents)
    unique_words = set(word for doc in documents for word in doc)
    idf_values = {}
    
    for word in unique_words:
        df_count = sum(1 for doc in documents if word in doc)
        idf_values[word] = np.log(N / (df_count + 1)) + 1
    
    return idf_values

# Calcular la TF (frecuencia de términos) para un documento
def compute_tf(document):
    word_count = len(document)
    if word_count == 0:
        return {}
    
    counter = Counter(document)
    return {word: count / word_count for word, count in counter.items()}

# Calcular la similitud coseno entre dos documentos usando Bag of Words
def cosine_bow(doc1, doc2):
    vocab = sorted(list(set(doc1) | set(doc2)))
    counter_doc1 = Counter(doc1)
    counter_doc2 = Counter(doc2)
    
    vec1 = [counter_doc1.get(word, 0) for word in vocab]
    vec2 = [counter_doc2.get(word, 0) for word in vocab]
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

# Calcular la similitud coseno entre dos documentos usando TF-IDF
def cosine_tfidf(doc1, doc2, idf_dict):
    tf_doc1 = compute_tf(doc1)
    tf_doc2 = compute_tf(doc2)
    
    vocab = sorted(list(set(doc1) | set(doc2)))
    
    tfidf1 = [tf_doc1.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
    tfidf2 = [tf_doc2.get(word, 0) * idf_dict.get(word, 0) for word in vocab]
    
    dot = np.dot(tfidf1, tfidf2)
    norm1 = np.linalg.norm(tfidf1)
    norm2 = np.linalg.norm(tfidf2)
    
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

# Calcular la similitud usando cadenas de Markov entre dos documentos
def markov_similarity(doc1, doc2):
    # Si alguno de los documentos es demasiado corto, no podemos crear una matriz de transición
    if len(doc1) <= 1 or len(doc2) <= 1:
        return 0
    
    all_tokens = sorted(list(set(doc1 + doc2)))
    token_index = {token: i for i, token in enumerate(all_tokens)}
    n = len(all_tokens)
    
    # Crear matrices de transición
    mat1 = np.zeros((n, n))
    mat2 = np.zeros((n, n))
    
    # Llenar matriz para doc1
    for i in range(len(doc1)-1):
        current = token_index[doc1[i]]
        next_token = token_index[doc1[i+1]]
        mat1[current][next_token] += 1
    
    # Normalizar filas de mat1
    row_sums = mat1.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Evitar división por cero
    mat1 = mat1 / row_sums[:, np.newaxis]
    
    # Llenar matriz para doc2
    for i in range(len(doc2)-1):
        current = token_index[doc2[i]]
        next_token = token_index[doc2[i+1]]
        mat2[current][next_token] += 1
    
    # Normalizar filas de mat2
    row_sums = mat2.sum(axis=1)
    row_sums[row_sums == 0] = 1
    mat2 = mat2 / row_sums[:, np.newaxis]
    
    # Aplanar matrices para calcular la similitud coseno
    flat1 = mat1.flatten()
    flat2 = mat2.flatten()
    
    dot = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

# Obtener el nivel de similitud basado en el valor del coseno
def get_similarity_level(value):
    if 0.85 <= value <= 1.00:
        return "high"
    elif 0.45 <= value < 0.85:
        return "medium"
    else:  # 0 <= value < 0.45
        return "low"

# Verificar si la detección de similitud es correcta
def is_correct_detection(expected_level, cosine_value):
    detected_level = get_similarity_level(cosine_value)
    return expected_level == detected_level

def find_texts_directory():
    """Busca la carpeta 'texts' en varias ubicaciones posibles"""
    # Posibles rutas para la carpeta de textos
    possible_paths = [
        "texts",                               
        os.path.join("Actividad 4.4", "texts"), 
        os.path.join("..", "texts"),             
        "."                                   
    ]
    
    for path in possible_paths:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "original.txt")):
            return path
    
    # Si no se encuentra la ruta, preguntar al usuario
    print("No se encontró automáticamente la carpeta 'texts'. Por favor, ingrese la ruta:")
    user_path = input().strip()
    if os.path.isdir(user_path):
        return user_path
    else:
        print(f"Error: La ruta '{user_path}' no es válida.")
        return None

def main():
    # Buscar y establecer la ruta de la carpeta de textos
    data_dir = find_texts_directory()
    
    if not data_dir:
        print("No se pudo encontrar la carpeta de textos. El programa se cerrará.")
        return
    
    # Leer archivo original
    original_file = "original.txt"
    original_path = os.path.join(data_dir, original_file)
    
    try:
        with open(original_path, 'r', encoding='utf-8') as file:
            original_text = file.read()
    except Exception as e:
        print(f"Error al leer el archivo original: {e}")
        return
    
    # Preprocesar el texto original
    original_tokens = preprocess(original_text)
    
    # Buscar archivos para comparar
    files_in_dir = [f for f in os.listdir(data_dir) if f.endswith(".txt") and f != original_file]
    
    # Filtrar para identificar los archivos similares (high, medium/moderate, low)
    similar_files = []
    for file in files_in_dir:
        if any(pattern in file.lower() for pattern in ["high", "medium", "moderate", "low"]):
            similar_files.append(file)
    
    if not similar_files:
        print("No se encontraron archivos similares para comparar.")
        return
    
    all_documents = [original_tokens]
    similar_tokens_dict = {}
    
    # Leer y preprocesar los otros archivos
    for similar_file in similar_files:
        try:
            with open(os.path.join(data_dir, similar_file), 'r', encoding='utf-8') as file:
                similar_text = file.read()
                similar_tokens = preprocess(similar_text)
                similar_tokens_dict[similar_file] = similar_tokens
                all_documents.append(similar_tokens)
        except Exception as e:
            print(f"Error al leer el archivo {similar_file}: {e}")
    
    # Calcular valores IDF para todos los documentos
    idf_dict = compute_idf(all_documents)
    
    # Diccionario para almacenar metadatos de similitud de los archivos
    similarities_metadata = {
        "similar_file": [],
        "expected_similarity": [],
        "bow_cosine": [],
        "is_bow_correct": [],
        "tfidf_cosine": [],
        "is_tfidf_correct": [],
        "markov_cosine": [],
        "is_markov_correct": []
    }
    
    # Determinar los niveles esperados según los nombres de los archivos
    expected_levels = {}
    for similar_file in similar_files:
        if "high" in similar_file.lower():
            expected_levels[similar_file] = "high"
        elif "moderate" in similar_file.lower() or "medium" in similar_file.lower():
            expected_levels[similar_file] = "medium"
        elif "low" in similar_file.lower():
            expected_levels[similar_file] = "low"
        else:
            print(f"No se pudo determinar el nivel esperado para: {similar_file}")
            expected_levels[similar_file] = "unknown"
    
    # Calcular similitudes para cada archivo similar
    for similar_file, similar_tokens in similar_tokens_dict.items():
        # Calcular similitud BOW
        bow_cosine = cosine_bow(original_tokens, similar_tokens)
        
        # Calcular similitud TF-IDF
        tfidf_cosine = cosine_tfidf(original_tokens, similar_tokens, idf_dict)
        
        # Calcular similitud Markov
        markov_cosine = markov_similarity(original_tokens, similar_tokens)
        
        # Obtener nivel esperado
        expected_level = expected_levels[similar_file]
        
        # Verificar si las detecciones son correctas
        is_bow_correct = is_correct_detection(expected_level, bow_cosine)
        is_tfidf_correct = is_correct_detection(expected_level, tfidf_cosine)
        is_markov_correct = is_correct_detection(expected_level, markov_cosine)
        
        # Guardar resultados
        similarities_metadata["similar_file"].append(similar_file)
        similarities_metadata["expected_similarity"].append(expected_level)
        similarities_metadata["bow_cosine"].append(bow_cosine)
        similarities_metadata["is_bow_correct"].append(is_bow_correct)
        similarities_metadata["tfidf_cosine"].append(tfidf_cosine)
        similarities_metadata["is_tfidf_correct"].append(is_tfidf_correct)
        similarities_metadata["markov_cosine"].append(markov_cosine)
        similarities_metadata["is_markov_correct"].append(is_markov_correct)
    
    # Crear DataFrame con los resultados
    df_results = pd.DataFrame(similarities_metadata)
    
    # Agregar columna con el nombre del archivo original
    df_results.insert(0, "original_file", original_file)
    
    # Calcular estadísticas de aciertos
    correct_bow = df_results["is_bow_correct"].sum()
    correct_tfidf = df_results["is_tfidf_correct"].sum()
    correct_markov = df_results["is_markov_correct"].sum()
    total_files = len(df_results)
    
    print(f"\nEstadísticas de aciertos:")
    print(f"BOW: {correct_bow}/{total_files} ({correct_bow/total_files:.2%})")
    print(f"TF-IDF: {correct_tfidf}/{total_files} ({correct_tfidf/total_files:.2%})")
    print(f"Markov: {correct_markov}/{total_files} ({correct_markov/total_files:.2%})")
    
    # Guardar resultados en CSV
    output_path = "comparison_results.csv"
    df_results.to_csv(output_path, index=False)
    
    # Generar resumen para el reporte
    print("\nResumen para el reporte:")
    print(f"Total de archivos comparados: {total_files}")
    
    # Identificar la técnica con mayor precisión
    techniques = {'BOW': correct_bow, 'TF-IDF': correct_tfidf, 'Markov': correct_markov}
    best_technique = max(techniques, key=techniques.get)
    print(f"Técnica con mayor precisión: {best_technique} ({techniques[best_technique]}/{total_files}, {techniques[best_technique]/total_files:.2%})")
    
    # Mostrar la distribución de niveles de similitud
    print(f"Distribución de niveles de similitud:")
    similarity_distribution = df_results["expected_similarity"].value_counts().to_dict()
    for level, count in similarity_distribution.items():
        print(f"  {level}: {count} archivos")

if __name__ == "__main__":
    main()