from pymongo import MongoClient
import pandas as pd

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bot_detection"]
users_collection = db["users"]

def load_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["bot_detection"]
    collection = db["users"]

    users_with_tweets = list(collection.find(
        {
            "tweets": {"$exists": True, "$ne": []},
            "label": {"$in": ["bot", "human"]}
        },
        {
            "user_id": 1, "label": 1, "user_creation": 1,
            "tweets": 1,
            "followers_count": 1, "following_count": 1,
            "tweet_count": 1, "verified": 1, "protected": 1,
            "profile_image_url": 1, "url": 1
        }
    ))

    data = []
    for user in users_with_tweets:
        tweets = [tweet["text"] for tweet in user["tweets"]]
        data.append({
            "user_id": user["user_id"],
            "label": user["label"],
            "user_creation": user["user_creation"],
            "tweet_sample": " ".join(tweets),
            "followers_count": user.get("followers_count", 0),
            "following_count": user.get("following_count", 0),
            "tweet_count": user.get("tweet_count", 0),
            "verified": int(user.get("verified", False)),
            "protected": int(user.get("protected", False)),
            "has_profile_image": int("default_profile" not in str(user.get("profile_image_url", ""))),
            "has_url": int(bool(user.get("url")))
        })

    df = pd.DataFrame(data)
    df["user_creation"] = pd.to_datetime(df["user_creation"], utc=True)
    df["user_age_days"] = (pd.Timestamp.now(tz='UTC') - df["user_creation"]).dt.days

    return df

def split_data(df, test_size=0.2):
    """Divide los datos en entrenamiento y prueba."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df["tweet_sample"], df["label"], test_size=test_size, random_state=42)