# Spam-email-Detection
This is a project that uses Machine learning and detect whether a mail received is spam or not based on the dataset provided to it. The language used here is Python.
The spam detector involves several key concepts and steps. Here's a detailed overview:

**1. Data Cleaning:**
Handling Missing Values: You dropped columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) containing a significant number of missing values.
Removing Duplicates: You removed duplicate entries from the dataset.
**2. Exploratory Data Analysis (EDA):**
Data Visualization: You created visualizations to explore the distribution of spam and ham messages.
Feature Engineering: You created new features such as the number of characters, words, and sentences in each text.
**3. Text Preprocessing:**
Lowercasing: Converted all text to lowercase.
Tokenization: Broke down sentences into individual words.
Removing Special Characters: Eliminated non-alphanumeric characters.
Removing Stopwords and Punctuation: Removed common English stopwords and punctuation.
Stemming: Reduced words to their root form using Porter stemming.
**4. Text Vectorization:**
Used the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert the transformed text into numerical features.
Set max_features=3000 to limit the number of features.
**5. Model Building:**
Employed various machine learning algorithms:
Naive Bayes:
Gaussian Naive Bayes (GNB): Used for continuous data.
Multinomial Naive Bayes (MNB): Suitable for discrete data like word counts.
Bernoulli Naive Bayes (BNB): Applied for binary data.
Ensemble Methods:
Voting Classifier: Combined predictions from multiple models (SVM, Naive Bayes, Extra Trees) using soft voting.
Stacking Classifier: Combined predictions from SVM, Naive Bayes, and Extra Trees using a Random Forest final estimator.
**6. Model Evaluation:**
Evaluated models using metrics such as accuracy, precision, and confusion matrix.
Considered different classifiers and ensemble methods to understand their performance.
**7. Model Improvement and Selection:**
Experimented with different variations, such as changing the max_features parameter in TF-IDF and using additional features like the number of characters.
Explored different classifiers and ensemble methods, comparing their performance.
**8. Deployment Preparation:**
Saved the TF-IDF vectorizer (vectorizer.pkl) and the Multinomial Naive Bayes model (model.pkl) using pickle for potential deployment.
**9. Concepts and Algorithms Used:**
Data Preprocessing: Cleaning, EDA, and Text Preprocessing.
Feature Engineering: Creating new features like character count, word count, and sentence count.
Text Vectorization: Converting text data into numerical features using TF-IDF.
Machine Learning Algorithms: Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Support Vector Machine (SVM), Decision Trees, Random Forest, AdaBoost, Gradient Boosting, XGBoost.
Ensemble Methods: Voting Classifier and Stacking Classifier.

Algorithms used are:- 
Gaussian Naive Bayes (GNB): Used for continuous data.
Multinomial Naive Bayes (MNB): Suitable for discrete data like word counts.
Bernoulli Naive Bayes (BNB): Applied for binary data.
Support Vector Machine (SVM): Utilized for classification tasks.
Decision Trees: Used for both classification and regression tasks.
Random Forest: An ensemble method based on decision trees.
AdaBoost: An ensemble method combining weak learners to create a strong learner.
Gradient Boosting: An ensemble method that builds trees sequentially to correct errors.
XGBoost: An optimized gradient boosting algorithm.
These algorithms were employed for classification tasks, and you also utilized ensemble methods such as the Voting Classifier and Stacking Classifier to combine predictions from multiple models. The specific algorithm that provided the best results for your spam classifier would depend on the dataset and the evaluation metrics used.

Modules used are:-
Streamlit :(Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science)
Pickle :(Pickle in Python is primarily used in serializing and deserializing a Python object structure. In other words, it's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.)
String :(string module contains a single utility function - capwords(s, sep=None). This function split the specified string into words using str. split(). Then it capitalizes each word using str)
nltk :( Natural Language Toolkit, is a Python package that you can use for NLP. A lot of the data that you could be analyzing is unstructured data and contains human-readable text. Before you can analyze that data programmatically, you first need to preprocess it.)


To Run the program use this :- streamlit run app.py in the terminal.
