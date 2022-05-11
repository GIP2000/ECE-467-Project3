#%%
import tensorflow as tf 
from random import shuffle
import numpy as np 
#import random 
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token = 'oov', filters = '"', split = ' ')
#%%
data = open("./data3.txt", 'rb').read().decode(encoding="utf-8")

corpus = data.lower().split("<te>")
corpus = [c + "<te>" for c in corpus]
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
#%%
input_seq = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_seq.append(n_gram_sequence)

max_seq_len = max([len(x) for x in input_seq])
input_seq = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_seq_len, padding='pre'))

# %%
xs = input_seq[:,:-1]
labels = input_seq[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
#%%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 200, input_length=max_seq_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
#%%
history = model.fit(xs, ys, epochs=50, verbose=1, batch_size=128)
model.save("saved_model/my_model2.h5")
# %%
seed = "obama"
import random 
while True:
    token_list = tokenizer.texts_to_sequences([seed])
    token_list = tf.keras.preprocessing.sequence.pad_sequences(token_list, maxlen = max_seq_len-1, padding='pre')
    probs = model.predict(token_list)

    probs = probs[0][1:]
    ind = random.choices(list(tokenizer.word_index.values()), weights = probs, k = 1000)
    ind = random.choice(ind)

    word = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(ind)]
    if word == "<te>":
      break
    seed += " " + word
print(seed)