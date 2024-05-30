import pandas as pd
import nltk
from cleanData import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

df = preprocess_data()

# Remove rows with null values in the "tweet" column
df = df.dropna(subset=["tweet"], inplace=False)
print(df.head(2))
df.info()
# Create new columns for storing
df['corpus'] = [nltk.word_tokenize(text) for text in df.tweet]
lemma = nltk.WordNetLemmatizer()
df['corpus'] = df.apply(lambda x: [lemma.lemmatize(word) for word in x.corpus], axis=1)
df['corpus'] = df.apply(lambda x: " ".join(x.corpus), axis=1)


print(df.head(5))
df.info()

# Split into train and validation data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Bag of Words (BoW)
count_vectorizer = CountVectorizer()
bow_train = count_vectorizer.fit_transform(train_df['corpus'])
bow_val = count_vectorizer.transform(val_df['corpus'])
bow = count_vectorizer.transform(df["corpus"])
# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(train_df['corpus'])
tfidf_val = tfidf_vectorizer.transform(val_df['corpus'])
tfidf = tfidf_vectorizer.transform(df['corpus'])
# Display the shapes of the transformed matrices
print("Bag of Words (BoW) matrix shape:", bow_train.shape)
print("TF-IDF matrix shape:", tfidf_train.shape)

# # Train logistic regression model on Bag of Words (BoW) representation
# log_reg_bow = LogisticRegression(max_iter=1000)
# log_reg_bow.fit(bow_train, train_df['Sentiment'])

# # Evaluate the model based on Bag of Words (BoW) representation
# val_pred_bow = log_reg_bow.predict(bow_val)
# accuracy_bow = accuracy_score(val_df['Sentiment'], val_pred_bow)
# print("Accuracy of logistic regression model (BoW):", accuracy_bow)

# # Train logistic regression model on TF-IDF representation
# log_reg_tfidf = LogisticRegression(max_iter=1000)
# log_reg_tfidf.fit(tfidf_train, train_df['Sentiment'])

# # Evaluate the model based on TF-IDF representation
# val_pred_tfidf = log_reg_tfidf.predict(tfidf_val)
# accuracy_tfidf = accuracy_score(val_df['Sentiment'], val_pred_tfidf)
# print("Accuracy of logistic regression model (TF-IDF):", accuracy_tfidf)

# # Define logistic regression models
# log_reg_bow = LogisticRegression(max_iter=1000)
# log_reg_tfidf = LogisticRegression(max_iter=1000)

# # Perform cross-validation on Bag of Words (BoW) representation
# bow_scores = cross_val_score(log_reg_bow, bow, df['Sentiment'], cv=5)
# print("Cross-Validation Scores (BoW):", bow_scores)
# print("Mean Accuracy (BoW):", bow_scores.mean())

# # Perform cross-validation on TF-IDF representation
# tfidf_scores = cross_val_score(log_reg_tfidf, tfidf, df['Sentiment'], cv=5)
# print("Cross-Validation Scores (TF-IDF):", tfidf_scores)
# print("Mean Accuracy (TF-IDF):", tfidf_scores.mean())

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],  # Optimization algorithm
}

# Initialize logistic regression model
log_reg = LogisticRegression(max_iter=1000)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to Bag of Words (BoW) representation
grid_search.fit(bow, df['Sentiment'])

# Fit GridSearchCV to TF-IDF representation
grid_search.fit(tfidf, df['Sentiment'])

# Save results to a text file
with open('grid_search_results.txt', 'w') as f:
    f.write("Best Parameters (BoW): {}\n".format(grid_search.best_params_))
    f.write("Best Score (BoW): {}\n".format(grid_search.best_score_))
    f.write("Grid Search CV Results (BoW):\n")
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        f.write("Parameters: {}, Mean Score: {:.4f}, Standard Deviation: {:.4f}\n".format(params, mean_score, std_score))

    f.write("\n\nBest Parameters (TF-IDF): {}\n".format(grid_search.best_params_))
    f.write("Best Score (TF-IDF): {}\n".format(grid_search.best_score_))
    f.write("Grid Search CV Results (TF-IDF):\n")
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        f.write("Parameters: {}, Mean Score: {:.4f}, Standard Deviation: {:.4f}\n".format(params, mean_score, std_score))


# Print best parameters and best score
print("Best Parameters (TF-IDF):", grid_search.best_params_)
print("Best Score (TF-IDF):", grid_search.best_score_)

