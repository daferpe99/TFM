from pymongo import MongoClient
import json

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bot_detection"]
users_collection = db["users"]

def insert_users(user_file_path):
    with open(user_file_path, "r", encoding="utf-8") as file:
        users = json.load(file)

    user_documents = []

    for user in users:
        user_id = user["id"] if user["id"].startswith("u") else f"u{user['id']}"

        user_data = {
            "user_id": user_id,
            "user_creation": user.get("created_at"),
            "description": user.get("description"),
            "location": user.get("location"),
            "name": user.get("name"),
            "username": user.get("username"),
            "profile_image_url": user.get("profile_image_url"),
            "verified": user.get("verified", False),
            "protected": user.get("protected", False),
            "url": user.get("url"),
            "public_metrics": user.get("public_metrics", {}),
        }

        # Convertir métricas públicas a enteros
        if "public_metrics" in user and isinstance(user["public_metrics"], dict):
            user_data["public_metrics"] = {
                "followers_count": int(user["public_metrics"].get("followers_count", 0)),
                "following_count": int(user["public_metrics"].get("following_count", 0)),
                "tweet_count": int(user["public_metrics"].get("tweet_count", 0)),
                "listed_count": int(user["public_metrics"].get("listed_count", 0))
            }

        # Limpiar valores None
        user_data = {k: v for k, v in user_data.items() if v is not None}

        print(f"📌 Procesando usuario: {user_id}")
        user_documents.append(user_data)

    # Insertar en MongoDB
    if user_documents:
        try:
            result = users_collection.insert_many(user_documents, ordered=False)
            print(f"✅ {len(result.inserted_ids)} usuarios insertados correctamente.")
        except Exception as e:
            print(f"❌ Error en insert_many: {e}")
    else:
        print("⚠️ No hay usuarios para insertar.")

# 📌 Llamar a la función para insertar los usuarios
insert_users("../../data/user.json")
