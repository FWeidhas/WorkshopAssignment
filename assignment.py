import os
import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

print("Downloading NLTK stopwords...")
# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
print("NLTK stopwords download complete.")


# Define a function to clean the text
def clean_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove # tags
    text = re.sub(r'#\w+', '', text)
    # Remove @ tags
    text = re.sub(r'@\w+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load or read CSV file
file_path = "./Corona_NLP_train_cleaned.csv"
if os.path.exists(file_path):
    print("Reading cleaned text data from file...")
    df = pd.read_csv(file_path)
else:
    print("Reading CSV file...")
    df = pd.read_csv("./Corona_NLP_train.csv", encoding='latin1')
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    print("Removing 'Location' and 'TweetAt' columns...")
    # Remove 'Location' and 'TweetAt' columns
    df = df.drop(columns=['Location', 'TweetAt'])
    print("Columns 'Location' and 'TweetAt' removed.")

    print("Removing empty values...")
    # Remove rows with any missing values
    df = df.dropna()
    print("Empty values removed.")
    print("Cleaning text data...")
    df['cleaned_tweet'] = df['OriginalTweet'].apply(clean_text)
    print("Text data cleaned.")
    print("Saving cleaned text data to file...")
    df.to_csv(file_path, index=False)
    print("Cleaned text data saved to file.")

print("Sampling 5% of the data...")
# Sample 5% of the data
df_sampled = df.sample(frac=0.05, random_state=42)
print("Data sampling complete.")

print("Vectorizing text data (Bag of Words)...")
# Bag of Words (BoW)
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df_sampled['cleaned_tweet'])
print("Text data vectorized using Bag of Words.")

print("Vectorizing text data (TF-IDF)...")
# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df_sampled['cleaned_tweet'])
print("Text data vectorized using TF-IDF.")

print("Performing K-means clustering...")
# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_tfidf)
print("K-means clustering complete.")


print("Performing Topic Modeling using Latent Dirichlet Allocation (LDA)...")
# Topic Modeling using Latent Dirichlet Allocation (LDA)
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
X_topics = lda_model.fit_transform(X_tfidf)
print("Topic Modeling using Latent Dirichlet Allocation (LDA) complete.")

print("Preparing data for training and evaluation...")
# Labels
y = df_sampled['Sentiment']

# Split data
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


# Train and evaluate model using Bag of Words (BoW)
print("Training and evaluating model using Bag of Words...")
model_bow = LogisticRegression()
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)
print("Accuracy (BoW):", accuracy_score(y_test, y_pred_bow))
print("Classification Report (BoW):\n", classification_report(y_test, y_pred_bow))

# Train and evaluate model using TF-IDF
print("Training and evaluating model using TF-IDF...")
model_tfidf = LogisticRegression()
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print("Accuracy (TF-IDF):", accuracy_score(y_test, y_pred_tfidf))
print("Classification Report (TF-IDF):\n", classification_report(y_test, y_pred_tfidf))
