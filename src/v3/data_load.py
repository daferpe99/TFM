import pandas as pd
import json
from sklearn.model_selection import train_test_split

#Carga los tweets y los asocia con el id de usuario y su label correspondiente
def load_data(tweet_path, label_path, user_path):
    # Cargar tweets
    with open(tweet_path, 'r') as tweet_file:
        tweets = json.load(tweet_file)
    tweet_df = pd.DataFrame(tweets)
    tweet_df['label_id'] = 'u' + tweet_df['author_id'].astype(str)
    tweet_df.rename(columns={"id": "tid", "created_at": "tweet_creation"}, inplace=True)

    # Consolidar tweets por usuario
    tweet_metrics = tweet_df.groupby('label_id').agg(
        #num_tweets=('tid', 'count'),                       # Número total de tweets
        #last_tweet=('tweet_creation', 'max'),              # Último tweet
        #avg_tweet_length=('text', lambda x: x.str.len().mean()),  # Longitud promedio de tweets
        tweets=('text', list)                              # Lista de tweets por usuario
    ).reset_index()

    # Cargar etiquetas
    label_df = pd.read_csv(label_path, header=0, names=["label_id", "label_value"])

    # Cargar datos de los usuarios
    with open(user_path, 'r', encoding='utf-8') as user_file:
        users = json.load(user_file)
    user_df = pd.DataFrame(users)
    user_df.rename(columns={"id": "label_id", "created_at": "user_creation"}, inplace=True)

    # Combinar datos de usuarios con métricas de tweets
    user_df = user_df.merge(tweet_metrics, on='label_id', how='left')
    user_df = user_df.merge(label_df, on='label_id', how='left')

    # Convertir fechas a datetime
    user_df['user_creation'] = pd.to_datetime(user_df['user_creation'])
    #user_df['last_tweet'] = pd.to_datetime(user_df['last_tweet'])

    # Filtrar columnas relevantes
    return user_df.rename(columns={"label_id": "user_id", "label_value": "label"})

def split_data(df, test_size=0.2):
    #Divide los datos en entrenamiento y prueba.
    return train_test_split(df['tweet_sample'], df['ordered_label'], test_size=test_size, random_state=42)

def load_user_data(user_path, ids):
    #Carga la información de los usuarios desde el archivo user.json
    with open (user_path, 'r') as user_file:
        users = json.load(user_file)
    user_df = pd.DataFrame(users)

    return user_df[user_df['id'].isin(ids)]