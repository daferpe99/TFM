import pandas as pd
import json
from sklearn.model_selection import train_test_split

#Carga los tweets y los asocia con el id de usuario y su label correspondiente
def load_tweet_data(tweet_path, label_path):
    """Carga tweets y etiquetas desde los archivos."""
    with open(tweet_path, 'r') as tweet_file:
        tweets = json.load(tweet_file)
    tweet_df = pd.DataFrame(tweets)
    label_df = pd.read_csv(label_path, header=0, names=["label_id", "label_value"])
    tweet_df['label_id'] = 'u' + tweet_df['author_id'].astype(str)
    merged_df = tweet_df.merge(label_df, on='label_id', how='left')
    return merged_df[["label_value", "text"]].rename(columns={"label_value": "ordered_label", "text": "tweet_sample"})

def split_data(df, test_size=0.2):
    """Divide los datos en entrenamiento y prueba."""
    return train_test_split(df['tweet_sample'], df['ordered_label'], test_size=test_size, random_state=42)