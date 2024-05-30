import os
import re
import pandas as pd
from nltk.corpus import stopwords

def preprocess_data(input_file="./Corona_NLP_train.csv", output_file="./Corona_NLP_train_cleaned.csv"):
    stopWords = set(stopwords.words('english'))

    # Check if the cleaned file already exists
    if os.path.exists(output_file):
        return pd.read_csv(output_file)

    # Read the data
    df = pd.read_csv(input_file, encoding='latin1')
    
    # Drop rows with any null values
    df.dropna(inplace=True)
    
    # Drop unnecessary columns
    df = df.drop(columns=["UserName", "Location", "TweetAt", "ScreenName"])
    
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Function to clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', " ", text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#\w+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z ]','', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.split()
        text = " ".join([word for word in text if word.lower() not in stopWords])

        return text

   
    # Apply text cleaning
    df["tweet"] = df["OriginalTweet"].apply(clean_text)

    # Drop the original tweet column
    df = df.drop(columns=["OriginalTweet"])

     # Drop rows with empty values after cleaning
    df.dropna(subset=['tweet'],inplace=False)

    # Save the cleaned dataframe
    df.to_csv(output_file, index=False)

    return df
