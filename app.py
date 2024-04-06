from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from functools import wraps
import pyrebase

config = {
    "apiKey": "AIzaSyBdboJz5Xoi30JyDT1pqxUJnCqWiyXKVIQ",
    "authDomain": "abstractivetextsummarization.firebaseapp.com",
    "databaseURL": "https://abstractivetextsummarization-default-rtdb.firebaseio.com/",
    "projectId": "abstractivetextsummarization",
    "storageBucket": "abstractivetextsummarization.appspot.com",
    "messagingSenderId": "985957501888",
    "appId": "1:985957501888:web:8fcfb59e63ba7c72c35625",
    "measurementId": "G-B336RKSZK7",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Model Starts Here....

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

logger = tf.get_logger()

class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        logger.debug(f"encoder_out_seq.shape = {encoder_out_seq.shape}")
        logger.debug(f"decoder_out_seq.shape = {decoder_out_seq.shape}")

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            logger.debug("Running energy computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_full_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            logger.debug(f"U_a_dot_h.shape = {U_a_dot_h.shape}")

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            logger.debug(f"Ws_plus_Uh.shape = {Ws_plus_Uh.shape}")

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            logger.debug(f"ei.shape = {e_i.shape}")

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            logger.debug("Running attention vector computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_full_seq * K.expand_dims(inputs, -1), axis=1)

            logger.debug(f"ci.shape = {c_i.shape}")

            return c_i, [c_i]

        # we don't maintain states between steps when computing attention
        # attention is stateless, so we're passing a fake state for RNN step function
        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e], constants=[encoder_out_seq]
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c], constants=[encoder_out_seq]
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


import numpy as np 
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Attention, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings


from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

with open('Summarized_Model//contraction_mapping.pkl', 'rb') as handle:
    contraction_mapping = pickle.load(handle)

with open('Summarized_Model//x_tokenizer.pickle', 'rb') as handle:
    X_tokenizer = pickle.load(handle)

with open('Summarized_Model//y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

model = load_model("Summarized_Model//encoder_model.h5", custom_objects={'AttentionLayer': AttentionLayer})


with open('Summarized_Model//model_info.pickle', 'rb') as handle:
    loaded_model_info = pickle.load(handle)

max_text_len = loaded_model_info['max_text_len']
max_summary_len = loaded_model_info['max_summary_len']
reverse_target_word_index = loaded_model_info['reverse_target_word_index']
reverse_source_word_index = loaded_model_info['reverse_source_word_index']
target_word_index = loaded_model_info['target_word_index']


encoder_model = load_model("Summarized_Model//encoder_model.h5", custom_objects={'AttentionLayer': AttentionLayer})

decoder_model = load_model("Summarized_Model//decoder_model.h5", custom_objects={'AttentionLayer': AttentionLayer})

def decode_sequence(input_seq):

    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1))

    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'//([^)]*//)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s//b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    if(num==0):
      tokens = [w for w in newString.split() if not w in stop_words]
    else :
      tokens = newString.split()
    long_words=[]
    for i in tokens:
            long_words.append(i)
    return (" ".join(long_words)).strip()


def summarize_text(input_text):
    cleaned_text = text_cleaner(input_text, 0)

    input_seq = X_tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(input_seq, maxlen=max_text_len, padding='post')

    predicted_summary = decode_sequence(padded_seq)

    return predicted_summary

# Model Ends Here....

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("sign.html")
    else:
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        email = request.form["email"]
        password = request.form["password"]
        try:
            auth.create_user_with_email_and_password(email, password)

            data = {"first_name": first_name, "last_name": last_name, "email": email}
            db.child("users").push(data)

            return redirect(url_for("abstractive-summarization"))
        except:
            message = "Email Already Exists"
            return render_template("sign.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    else:
        email = request.form["email"]
        password = request.form["password"]
        try:
            auth.sign_in_with_email_and_password(email, password)

            session["user"] = email

            return redirect(url_for("abstractive_summarization"))
        except:
            message = "Invalid Email or Password"
            return render_template("login.html", message=message)


@app.route("/abstractive-summarization", methods=["GET", "POST"])
@login_required
def abstractive_summarization():
    return render_template("summarization.html")


@app.route("/reset", methods=["GET", "POST"])
def reset():
    if request.method == "GET":
        return render_template("reset.html")
    else:
        email = request.form["email"]
        try:
            auth.send_password_reset_email(email)
            message = "An email to reset the password has been successfully sent"
            return render_template("reset.html", message=message)
        except:
            message = "Something went wrong. Please check if the email you provided is registered or not."
            return render_template("reset.html", message=message)
        
@app.route('/get_summarized_data', methods=['POST'])
def get_summarized_data():
    input_text = request.form['input_text']
    summarized_text = summarize_text(input_text)
    return jsonify({'summarized_text': summarized_text})


if __name__ == "__main__":
    app.run(debug=True)
