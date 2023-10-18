import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tenses_list = ['Simple Past', 'Past Continuous ', 'Past Perfect', 'Future Continuous ', 'Simple Future', 'Future Perfect', 'Simple Present', 'Past Perfect Continuous', 'Future Perfect Continuous', 'Present Perfect Continuous', 'Present Continuous', 'Present Perfect']

def get_model():
    dataset = pd.read_excel('./tense.xlsx')
    dataset.info()
    dataset.sample(3)
    # dataset['Tense'].value_counts()
    
    train_sentences, rem_sentences, train_tenses, rem_tenses = train_test_split(dataset['Sentences'], dataset['Tense'], train_size = 0.80)
    val_sentences, test_sentences, val_tenses, test_tenses = train_test_split(rem_sentences, rem_tenses, test_size=0.1)

    tenses_list = ['Simple Past', 'Past Continuous ', 'Past Perfect', 'Future Continuous ', 'Simple Future', 'Future Perfect', 'Simple Present', 'Past Perfect Continuous', 'Future Perfect Continuous', 'Present Perfect Continuous', 'Present Continuous', 'Present Perfect']
    train_labels = train_tenses.replace(
        to_replace=tenses_list,
        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )

    test_labels = test_tenses.replace(
        to_replace=tenses_list,
        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )

    val_labels = val_tenses.replace(
        to_replace=tenses_list,
        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )

    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=12000,
        output_mode='int',
        output_sequence_length=15
    )

    text_vectorizer.adapt(train_sentences)

    #LSTM
    embedding_2 = tf.keras.layers.Embedding(
        input_dim=len(text_vectorizer.get_vocabulary()),
        output_dim=128,
        embeddings_initializer='uniform',
        input_length=30,
        name='embedding_2'
    )
    #Input Layer
    inputs = tf.keras.layers.Input(shape=(1,), dtype='string')
    #Hidden layer
    x = text_vectorizer(inputs)
    x = embedding_2(x)
    x = tf.keras.layers.LSTM(64)(x)
    #Output layer
    outputs = tf.keras.layers.Dense(12, activation='softmax')(x)

    model_2 = tf.keras.Model(inputs, outputs, name='model_2_simple_lstm')
    model_2.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model_2_history = model_2.fit(
        train_sentences,
        train_labels,
        epochs=2,
        validation_data=(val_sentences, val_labels)
    )
    return model_2
st.title('Grammar Dectection')
sentence = st.text_input("Enter a Sentence")
model = get_model()
st.subheader("Result", anchor=None, help=None, divider=True)
result_dict = {}
prediction = model.predict([sentence])[0]
for index, result in enumerate(prediction):
  result_dict[tenses_list[index]] = result
result_dict = sorted(result_dict.items(), key=lambda x:x[1], reverse=True)
if(result_dict):
  for key in result_dict:
    st.progress(int(key[1]*100), text=key[0])
  # print(f"{key[0]} --> {key[1]*100} %")
# st.progress(90, text=None)

