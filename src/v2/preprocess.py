import string
import nltk
from nltk.corpus import stopwords
from sklearn.utils import resample
import pandas as pd

nltk.download('stopwords')

def downsampling_data(data):
    # DOWNSAMPLING
    # Separar las clases en dos DataFrames: clase mayoritaria y clase minoritaria
    majority_class = data[data['ordered_label'] == 'human']
    minority_class = data[data['ordered_label'] == 'bot']

    # Aplicar downsampling a la clase mayoritaria
    majority_downsampled = resample(
        majority_class,
        replace=False,                 # Sin reemplazo
        n_samples=len(minority_class), # Igualar el número de muestras de la clase minoritaria
        random_state=42                # Para reproducibilidad
    )

    # Combinar el downsampled de la clase mayoritaria con la clase minoritaria
    downsampled_data = pd.concat([majority_downsampled, minority_class])

    # Opcional: mezclar el dataset para eliminar cualquier patrón
    downsampled_data = downsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Mostrar el conteo de etiquetas después del downsampling
    print(downsampled_data['ordered_label'].value_counts())


    # Convertir 'ordered_label' en binario: 'human' -> 0 y 'bot' -> 1
    downsampled_data['ordered_label'] = downsampled_data['ordered_label'].map({'human': 0, 'bot': 1})

    return downsampled_data

def remove_punctuations(text):
    """Elimina signos de puntuación de un texto."""
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    """Elimina palabras vacías del texto."""
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])