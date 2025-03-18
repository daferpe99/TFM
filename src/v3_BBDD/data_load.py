from pymongo import MongoClient
import pandas as pd

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bot_detection"]
users_collection = db["users"]

def load_data():
    """Carga los datos de los usuarios y sus tweets desde MongoDB."""
    users = list(users_collection.find({"tweets": {"$exists": True, "$not": {"$size": 0}}}))  # Solo usuarios con tweets
    
    # Transformar datos en DataFrame
    df = pd.DataFrame(users)

    # Asegurar que los datos esenciales estén presentes
    df["user_creation"] = pd.to_datetime(df["user_creation"], errors="coerce")
    df["label"] = df["label"].astype(str)

    # Convertir lista de tweets en texto unido
    df["tweet_sample"] = df["tweets"].apply(lambda x: " ".join(tweet["text"] for tweet in x))

    return df

def split_data(df, test_size=0.2):
    """Divide los datos en entrenamiento y prueba."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df["tweet_sample"], df["label"], test_size=test_size, random_state=42)