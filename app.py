from flask import Flask, request, render_template, send_file
import pandas as pd
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob

nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

last_prediction_df = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction_df
    input_text = request.form.get('text', '')
    file = request.files.get('file')

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        if 'text' not in df.columns:
            return render_template('index.html', error="CSV must contain a 'text' column.")

        df['Cleaned Text'] = df['text'].astype(str).str.lower().apply(correct_spelling)
        df['VADER Sentiment'] = df['Cleaned Text'].apply(get_vader_sentiment)
        vader_scores = df['Cleaned Text'].apply(vader_analyzer.polarity_scores)

        df['Positive %'] = vader_scores.apply(lambda x: round(x['pos'] * 100, 2))
        df['Negative %'] = vader_scores.apply(lambda x: round(x['neg'] * 100, 2))
        df['Neutral %'] = vader_scores.apply(lambda x: round(x['neu'] * 100, 2))

        sentiment_counts = df['VADER Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
        create_pie_chart(sentiment_counts)

        last_prediction_df = df
        return render_template('index.html', table=df.to_html(classes='data', header="true"), show_download=True, show_chart=True)

    elif input_text.strip() != '':
        cleaned_text = correct_spelling(input_text.lower())
        scores = vader_analyzer.polarity_scores(cleaned_text)
        vader_sentiment = get_vader_sentiment(cleaned_text)

        pos_percent = round(scores['pos'] * 100, 2)
        neg_percent = round(scores['neg'] * 100, 2)
        neu_percent = round(scores['neu'] * 100, 2)

        last_prediction_df = pd.DataFrame({
            'Original Text': [input_text],
            'Corrected Text': [cleaned_text],
            'VADER Sentiment': [vader_sentiment],
            'Positive %': [pos_percent],
            'Neutral %': [neu_percent],
            'Negative %': [neg_percent]
        })

        pie_data = pd.Series([pos_percent, neu_percent, neg_percent], index=['Positive', 'Neutral', 'Negative'])
        create_pie_chart(pie_data)

        return render_template(
            'index.html',
            prediction=vader_sentiment,
            text=input_text,
            corrected_text=cleaned_text,  # ✅ added this line
            pos_percent=f"{pos_percent}%",
            neu_percent=f"{neu_percent}%",
            neg_percent=f"{neg_percent}%",
            show_download=True,
            show_chart=True
        )

    return render_template('index.html', error="Please enter text or upload a CSV file.")

@app.route('/download')
def download():
    global last_prediction_df
    if last_prediction_df is not None:
        csv_path = "output.csv"
        last_prediction_df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True)
    return "No prediction made yet to download."

def get_vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.35:
        return 'Positive'
    elif compound <= -0.35:
        return 'Negative'
    else:
        return 'Neutral'

def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text

def create_pie_chart(data):
    plt.figure(figsize=(5, 5))
    colors = ['limegreen', 'gray', 'tomato']
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140,
            colors=colors, textprops={'color': "blue"})
    plt.title('VADER Sentiment Distribution', color='white')
    plt.tight_layout()
    plt.gca().set_facecolor('#222')
    plt.savefig('static/pie_chart.png', transparent=True)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
