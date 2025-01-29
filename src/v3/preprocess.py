import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.utils import resample
from contractions import fix
import pandas as pd
import re
from datetime import datetime
from pytz import utc
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords') #Descarga la lista de stopwords
nltk.download('wordnet')  #Descarga recursos necesarios para lematización
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def downsampling_data(data):
    # DOWNSAMPLING
    # Separar las clases en dos DataFrames: clase mayoritaria y clase minoritaria
    majority_class = data[data['label'] == 'human']
    minority_class = data[data['label'] == 'bot']

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
    print(downsampled_data['label'].value_counts())


    # Convertir 'ordered_label' en binario: 'human' -> 0 y 'bot' -> 1
    downsampled_data['label'] = downsampled_data['label'].map({'human': 0, 'bot': 1})

    return downsampled_data

#PREPROCESS TEXT
def remove_punctuations(text):
    #Elimina signos de puntuación de un texto.
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    #Elimina palabras vacías del texto, excepto not nor y but
    stop_words = set(stopwords.words('english')) - {'not', 'nor', 'but'}
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    #Lemmatiza las palabras del texto
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def stem_text(text):
    #Aplica stemming a las palabras del texto
    return " ".join([stemmer.stem(word) for word in text.split()])

def stem_and_lemmatize(text):
    #Combina la lemmatización con el stemming del texto:
    lemmatized = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    stem = " ".join([stemmer.stem(word) for word in lemmatized.split()])

    return stem

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) #Elimina las urls
    text = re.sub(r"@\w+", "", text) #Elimina las menciones
    text = re.sub(r"#\w+", "", text) #Elimina los hashtags

    return text

def extract_mentions_hash(text):
    #Extrae menciones y hashtags del texto para posterior uso
    mentions = re.findall(r"@\w+", text)
    hashtags = re.findall(r"#\w+", text)

    return mentions, hashtags

def to_lowercase(text):
    #Pasamos todo el texto a minusculas
    return text.lower()

def expand_contractions(text):
    #Eliminamos las contracciones
    return fix(text)

def remove_emojis(text):
    """Elimina emojis del texto."""
    return text.encode('ascii', 'ignore').decode('ascii')

def normalize_text(text):
    """Normaliza el texto para garantizar que esté en UTF-8."""
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return ''.join([char for char in text if ord(char) < 128])

def preprocess_tweets(tweet_list):
    """Preprocesa una lista de tweets."""
    if isinstance(tweet_list, list):
        return [normalize_text(clean_tweet(tweet)) for tweet in tweet_list]
    return []

#PREPRCESS CREATION DATE
def preprocess_creation_date(data):
    scaler = MinMaxScaler()
    data["user_creation"] = pd.to_datetime(data["user_creation"], utc=True)
    # Calcular antigüedad en días
    data['age_days'] = (datetime.now(utc) - data['user_creation']).dt.days
    data['age_days_scaled'] = scaler.fit_transform(data[['age_days']])
    return data

