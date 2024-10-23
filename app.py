# app.py

from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Clear any previous TensorFlow sessions and load your model
K.clear_session()
model = load_model('cnn_model.h5')
print("Model loaded successfully")

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    le = pickle.load(handle)

# Initialize Flask app
app = Flask(__name__)

# Home route to display input form
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['tweet_text']

        # Error Handling
        if not user_input.strip():
            return render_template('index.html', warning="Please enter a tweet to analyze.")

        # Predict sentiment
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        prediction = model.predict(padded_sequence)
        predicted_class = prediction.argmax(axis=1)
        sentiment = le.inverse_transform(predicted_class)[0]

        # Create probability DataFrame
        sentiment_labels = le.classes_
        prob_df = pd.DataFrame({
            'Sentiment': sentiment_labels,
            'Probability': prediction[0]
        }).sort_values(by='Probability', ascending=False)

        # Plot probabilities using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Sentiment', y='Probability', data=prob_df, ax=ax)
        ax.set_title('Probability of Each Sentiment Class')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Probability')

        # Save plot to static folder
        if not os.path.exists('static'):
            os.makedirs('static')
        fig_path = os.path.join('static', 'sentiment_plot.png')
        plt.savefig(fig_path)
        plt.close()

        return render_template('result.html', tweet_text=user_input, sentiment=sentiment, probabilities=prob_df, fig_path=fig_path)

if __name__ == '__main__':
    app.run(debug=True)
