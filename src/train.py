import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications import DenseNet201
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Reshape, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import os

# Load the captions
data = pd.read_csv('dataset/captions.txt')

# Clean captions (remove startseq and endseq if they exist)
data['caption'] = data['caption'].apply(lambda x: x.replace("startseq", "").replace("endseq", "").strip())

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['caption'])
vocab_size = len(tokenizer.word_index) + 1

# Save the tokenizer
with open('models/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate max_caption_length
max_caption_length = max(len(caption.split()) for caption in data['caption']) + 1

# Feature extraction using DenseNet201
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Dense(1920, activation='relu')(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)
feature_extractor.save('models/feature_extractor.keras')

# Extract features for all images
features = {}
from keras.preprocessing.image import load_img, img_to_array

for img_name in data['image'].unique():
    img_path = os.path.join('dataset/Images', img_name)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img, verbose=0)
    features[img_name] = feature

# Save features (optional, for faster loading later)
with open('models/features.pkl', 'wb') as f:
    pickle.dump(features, f)

# Prepare sequences for training
def create_sequences(tokenizer, max_length, captions, features, vocab_size):
    X1, X2, y = [], [], []
    for img_name, group in captions.groupby('image'):
        caption_list = group['caption'].tolist()
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = np.zeros(vocab_size)
                out_seq[out_seq] = 1
                X1.append(features[img_name][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_sequences(tokenizer, max_caption_length, data, features, vocab_size)

# Split data
from sklearn.model_selection import train_test_split
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Define the model
input1 = Input(shape=(1920,))
input2 = Input(shape=(max_caption_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped, sentence_features], axis=1)
sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
caption_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Callbacks
checkpoint = ModelCheckpoint(
    'models/model.keras',
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

earlystopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=1,
    restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    verbose=1,
    factor=0.2,
    min_lr=0.00000001
)

# Train the model
history = caption_model.fit(
    [X1_train, X2_train], y_train,
    epochs=10,
    validation_data=([X1_val, X2_val], y_val),
    callbacks=[checkpoint, earlystopping, learning_rate_reduction]
)