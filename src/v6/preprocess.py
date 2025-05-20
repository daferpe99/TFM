import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
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
    bots = data[data["label"] == "bot"]
    humans = data[data["label"] == "human"]

    # Asegura que la clase mayoritaria se reduzca a tamaño de la minoritaria
    if len(bots) < len(humans):
        humans_downsampled = humans.sample(len(bots), random_state=42)
        data_balanced = pd.concat([bots, humans_downsampled])
    else:
        bots_downsampled = bots.sample(len(humans), random_state=42)
        data_balanced = pd.concat([bots_downsampled, humans])

    return data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

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
    if not isinstance(text, str):
        return ""

    try:
        # Normalizar texto a UTF-8 válido ignorando errores
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        # Eliminar URLs, menciones, hashtags
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)

        # Opcional: eliminar caracteres especiales o emojis
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Espacios innecesarios
        text = text.strip()

        return text
    except Exception as e:
        print(f"[⚠️ Error al limpiar texto]: {e}")
        return ""

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

# Preprocesamiento de metadatos
def preprocess_user_metrics(data):
    scaler = MinMaxScaler()
    data["user_creation"] = pd.to_datetime(data["user_creation"], utc=True)

    # Calcular antigüedad de la cuenta
    data['user_age_days'] = (datetime.now(utc) - data['user_creation']).dt.days
    
    # Transformar variables numéricas con log(1 + x)
    data['followers_count_log'] = np.log1p(data['followers_count'])
    data['following_count_log'] = np.log1p(data['following_count'])
    data['tweet_count_log'] = np.log1p(data['tweet_count'])

    # Evitar divisiones por cero y valores extremos en el ratio
    data['followers_following_ratio'] = data.apply(
        lambda row: row['followers_count'] / row['following_count'] if row['following_count'] > 0 else 0,
        axis=1
    )
    data['followers_following_ratio_log'] = np.log1p(data['followers_following_ratio'])

    # Tweets por día (actividad)
    data['tweets_per_day'] = data.apply(
        lambda row: row['tweet_count'] / row['user_age_days'] if row['user_age_days'] > 0 else 0,
        axis=1
    )
    data['tweets_per_day_log'] = np.log1p(data['tweets_per_day'])

    # Escalar todas las variables transformadas
    cols_to_scale = [
        "followers_count_log",
        "following_count_log",
        "tweet_count_log",
        "user_age_days",
        "followers_following_ratio_log",
        "tweets_per_day_log"
    ]

    for col in cols_to_scale:
        data[col + "_scaled"] = scaler.fit_transform(data[[col]])

    return data



