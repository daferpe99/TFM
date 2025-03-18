import pandas as pd
from pymongo import MongoClient, UpdateOne

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bot_detection"]
users_collection = db["users"]

# Cargar el archivo CSV de etiquetas
label_df = pd.read_csv("../../data/label.csv", header=0, names=["user_id", "label"])

# Crear operaciones de actualización en lote
bulk_operations = []
for _, row in label_df.iterrows():
    update_query = {"user_id": row["user_id"]}
    update_data = {"$set": {"label": row["label"]}}

    bulk_operations.append(UpdateOne(update_query, update_data, upsert=False))

# Ejecutar la actualización en MongoDB
if bulk_operations:
    result = users_collection.bulk_write(bulk_operations)
    print(f"✅ {result.modified_count} usuarios actualizados con su etiqueta de 'label'.")
else:
    print("⚠️ No se encontraron usuarios para actualizar.")
