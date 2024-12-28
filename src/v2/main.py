from data_load import load_tweet_data, split_data
from preprocess import remove_punctuations, remove_stopwords, downsampling_data, lemmatize_text, extract_mentions_hash, clean_tweet, to_lowercase, expand_contractions, stem_text, stem_and_lemmatize
from model import build_model
from training import train_model
from evaluation import evaluate_model, save_results
from visualization import plot_samples, plot_training_history
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split



# Carga y preprocesa los datos
data = load_tweet_data('../../data/tweet.json', '../../data/label.csv')
print(data.head())
plot_samples(data)

#Donwsampling de los datos
data = downsampling_data(data)
plot_samples(data)

#Almacenamos las menciones y los hashtags de los tweets
data['mentions'], data['hashtags'] = zip(*data['tweet_sample'].apply(extract_mentions_hash))
#Eliminamos URLS, menciones y hashtags del texto de los tweets
data['tweet_sample_clean'] = data['tweet_sample'].apply(clean_tweet)

#Usando lemmatización del texto:
#data['tweet_sample_clean'] = data['tweet_sample_clean'].apply(to_lowercase).apply(expand_contractions).apply(remove_punctuations).apply(remove_stopwords).apply(lemmatize_text)
#Usando stemming del texto:
#data['tweet_sample_clean'] = data['tweet_sample_clean'].apply(to_lowercase).apply(expand_contractions).apply(remove_punctuations).apply(remove_stopwords).apply(stem_text)
#Combinacion de stemming y lematizacion:
data['tweet_sample_clean'] = data['tweet_sample_clean'].apply(to_lowercase).apply(expand_contractions).apply(remove_punctuations).apply(remove_stopwords).apply(stem_and_lemmatize)

train_X, test_X, train_Y, test_Y = train_test_split(data['tweet_sample_clean'],
                                                    data['ordered_label'],
                                                    test_size=0.2,
                                                    random_state=42)

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=100)
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=100)

# Construcción 
input_dim = len(tokenizer.word_index) + 1
model = build_model(input_dim=input_dim)
model.summary()
#Compilación del modelo
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy'],
    optimizer='adam'
)

#Entramiento del modelo
history = train_model(model, train_sequences, train_Y, test_sequences, test_Y)
plot_training_history(history)

# Evaluación
loss, accuracy = evaluate_model(model, test_sequences, test_Y)

#save_results("../../resultados/v2_lematizacion_&_stemming.txt",model=model, loss=loss, accuracy=accuracy, model_name="Modelo LSTM con lematizacion y stemming")

