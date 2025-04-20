import json
import os
from tqdm import tqdm

# Configura tus rutas y tamaño de fragmento
archivo_entrada = "../../data/tweets/tweet_2.json" 
directorio_salida = "../../data/tweets/tweet_2" 
tweets_por_fragmento = 10000  # Puedes ajustar este número

# Crear carpeta si no existe
os.makedirs(directorio_salida, exist_ok=True)

# Cargar todos los tweets
with open(archivo_entrada, "r", encoding="utf-8") as f:
    datos = json.load(f)

total = len(datos)
num_fragmentos = (total + tweets_por_fragmento - 1) // tweets_por_fragmento
print(f"📦 Total de tweets: {total} | Fragmentos esperados: {num_fragmentos}")

# Fragmentar con barra de progreso
for i in tqdm(range(0, total, tweets_por_fragmento), desc="✂️ Fragmentando", unit="fragmento"):
    parte = datos[i:i + tweets_por_fragmento]
    nombre_archivo = os.path.join(directorio_salida, f"parte_{i // tweets_por_fragmento:03d}.json")
    with open(nombre_archivo, "w", encoding="utf-8") as salida:
        json.dump(parte, salida, ensure_ascii=False)

print("🎉 Fragmentación completada.")