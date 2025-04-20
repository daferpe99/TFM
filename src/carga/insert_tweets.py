import json
import os
from pymongo import MongoClient, UpdateOne
from bs4 import BeautifulSoup

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bot_detection"]
users_collection = db["users"]


def clean_text(text):
    """Limpia el texto y lo convierte a UTF-8."""
    if text:
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return ""

def process_tweet(tweet):
    """Procesa cada tweet asegurando tipos de datos correctos."""
    public_metrics = tweet.get("public_metrics", {})

    return {
        "tweet_id": f"t{tweet.get('id', '')}",
        "text": clean_text(tweet.get("text", "")),
        "tweet_creation": tweet.get("created_at", ""),
        "lang": tweet.get("lang", ""),
        "possibly_sensitive": tweet.get("possibly_sensitive", False),
        "conversation_id": str(tweet["conversation_id"]) if tweet.get("conversation_id") else None,
        "in_reply_to_user_id": f"u{tweet['in_reply_to_user_id']}" if tweet.get("in_reply_to_user_id") else "unknown",
        "reply_settings": tweet.get("reply_settings", "everyone"),
        "retweet_count": int(public_metrics.get("retweet_count", 0) or 0),
        "reply_count": int(public_metrics.get("reply_count", 0) or 0),
        "like_count": int(public_metrics.get("like_count", 0) or 0),
        "quote_count": int(public_metrics.get("quote_count", 0) or 0),
        "source": BeautifulSoup(tweet.get("source", ""), "html.parser").text.strip() if tweet.get("source") else "",
        "mentions": [
            {"screen_name": mention["screen_name"], "user_id": f"u{mention['id']}"}
            for mention in tweet.get("entities", {}).get("user_mentions", [])
        ],
        "hashtags": [
            {"text": tag["text"]}
            for tag in tweet.get("entities", {}).get("hashtags", [])
        ],
        "media": [
            {"media_id": str(media["id"]), "media_url": media["media_url"], "type": media["type"]}
            for media in tweet.get("entities", {}).get("media", [])
        ] if "media" in tweet.get("entities", {}) else []
    }

def insert_tweets_from_files(tweet_folder_path):
    """Procesa archivos de tweets y los inserta en la BD evitando duplicados."""
    batch_size = 1000
    log_path = os.path.join(tweet_folder_path, "archivos_ingestados.txt")

    for file_name in sorted(os.listdir(tweet_folder_path)):
        if not file_name.endswith(".json"):
            continue  # Evita procesar archivos como el log

        file_path = os.path.join(tweet_folder_path, file_name)
        print(f"\n📂 Procesando archivo: {file_name}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                tweets = json.load(file)
        except Exception as e:
            print(f"❌ Error al leer el archivo {file_name}: {e}")
            continue

        batch = []
        for tweet in tweets:
            tweet_data = process_tweet(tweet)
            user_id = f"u{tweet.get('author_id', '')}"

            user_exists = users_collection.find_one({"user_id": user_id}, {"_id": 1})
            if not user_exists:
                print(f"⚠️ Usuario {user_id} no encontrado en la BD. Creando nuevo registro.")
                users_collection.insert_one({"user_id": user_id, "tweets": []})

            existing_tweet = users_collection.find_one(
                {"user_id": user_id, "tweets.tweet_id": tweet_data["tweet_id"]},
                {"tweets.$": 1}
            )
            if existing_tweet:
                print(f"🔴 Tweet {tweet_data['tweet_id']} ya está en la BD. Omitiendo...")
                continue

            update_query = {"user_id": user_id}
            update_data = {"$addToSet": {"tweets": tweet_data}}
            batch.append(UpdateOne(update_query, update_data))

            if len(batch) >= batch_size:
                bulk_insert(batch)
                batch = []

        if batch:
            bulk_insert(batch)

        # ✅ Eliminar archivo y registrar en log
        try:
            os.remove(file_path)
            print(f"🗑️ Archivo {file_name} eliminado tras ingesta exitosa.")

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(file_name + "\n")

        except Exception as e:
            print(f"❌ Error al eliminar el archivo {file_name}: {e}")
def bulk_insert(batch):
    """Realiza la inserción en MongoDB y maneja errores."""
    try:
        result = users_collection.bulk_write(batch)
        print(f"✅ Lote insertado con {len(batch)} operaciones")
        print(f"🔎 Verificación: Se han modificado {result.modified_count} documentos.")
    except Exception as e:
        print(f"\n❌ Error en bulk_write: {e}")
        with open("error_log.txt", "a", encoding="utf-8") as error_log:
            error_log.write(f"Error: {e}\n")
            for op in batch:
                error_log.write(json.dumps(op._doc, indent=2) + "\n")
            error_log.write("\n\n")

# Llamar a la función con la carpeta que contiene los tweets
insert_tweets_from_files("../../data/tweets/tweet_1")
