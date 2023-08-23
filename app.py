import gradio as gr

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Embedding
from tensorflow.keras.layers import TextVectorization

path = '/vectorizer'
path_m = '/model'

def return_vec(path):
    vec_model = tf.keras.models.load_model(path)
    vectorizer = vec_model.layers[0]
    return vectorizer

vectorizer = return_vec(path)
model = tf.keras.models.load_model(path_m)


def score_comment(comment):
    target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    vec_com = vectorizer([comment])
    results = model.predict(vec_com)

    text = ''
    for idx, col in enumerate(target):
        text += '{}:{}\n'.format(col, results[0][idx]>0.5)

    return text

interface = gr.Interface(fn = score_comment, inputs = gr.inputs.Textbox(lines=2,placeholder = "type yout comment..."),outputs = 'text')
interface.launch(share=True)
