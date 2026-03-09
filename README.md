# Sentiment Analysis Web Application

A Flask-based web application that analyzes the sentiment of text using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) model.

## Features

- **Single Text Analysis**: Analyze sentiment of individual text inputs
- **Batch Processing**: Upload CSV files for bulk sentiment analysis
- **Sentiment Classification**: Classifies text as Positive, Negative, or Neutral
- **Sentiment Scores**: Shows percentage breakdown (Positive %, Neutral %, Negative %)
- **Spell Correction**: Automatically corrects spelling errors using TextBlob
- **Data Visualization**: Generates pie charts showing sentiment distribution
- **CSV Export**: Download results as CSV file

## Technology Stack

- **Backend**: Flask (Python web framework)
- **ML Model**: NLTK VADER (Sentiment Analysis)
- **Data Processing**: Pandas, NumPy
- **Text Processing**: TextBlob (Spell Correction)
- **Visualization**: Matplotlib (Pie Charts)
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
```bash
git clone https://github.com/codergopaldhali/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install flask pandas nltk textblob matplotlib werkzeug
```

## How to Run

1. Activate virtual environment:
```bash
venv\Scripts\activate
```

2. Run the application:
```bash
python app.py
```

3. Open browser and go to:
```
http://localhost:5000
```

## Project Structure
```
Sentiment-Analysis-App/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── static/               # CSS, JS, pie charts
├── uploads/              # CSV file uploads
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## How It Works

1. **Text Analysis**: 
   - User enters text or uploads CSV
   - Text is cleaned and spell-corrected
   - VADER analyzes sentiment using compound score
   - Results displayed with percentages

2. **VADER Sentiment Score**:
   - Compound >= 0.35: **Positive**
   - Compound <= -0.35: **Negative**
   - Between: **Neutral**

3. **Visualization**:
   - Pie chart shows sentiment distribution
   - Results can be exported as CSV

## Example Usage

**Input**: "I absolutely love this amazing product!"
**Output**: Positive (Positive: 85%, Neutral: 10%, Negative: 5%)

## Author

Gopal Dhali

## License
Your README content here