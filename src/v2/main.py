from data_load import load_tweet_data, split_data
from preprocess import remove_punctuations, remove_stopwords, downsampling_data
from model import build_model
from training import train_model
from evaluation import evaluate_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Carga y preprocesa los datos
data = load_tweet_data('../../data/tweet.json', '../../data/label.csv')
print(data.head())
sns.countplot(x='ordered_label', data=data)
plt.xlabel("Etiquetas")
plt.ylabel("Cantidad")
plt.show()


#Donwsampling de los datos
data = downsampling_data(data)
sns.countplot(x='ordered_label', data=data)
plt.xlabel("Etiquetas")
plt.ylabel("Cantidad")
plt.show()

data['tweet_sample'] = data['tweet_sample'].apply(remove_punctuations).apply(remove_stopwords)

train_X, test_X, train_Y, test_Y = train_test_split(data['tweet_sample'],
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

# Evaluación
evaluate_model(model, test_sequences, test_Y)
