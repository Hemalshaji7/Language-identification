import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk

# Set UTF-8 encoding
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Display the first 10 rows of the data
st.write("First 10 rows of the dataset:")
st.write(data.head(10))

# Check for missing values
st.write("Number of missing values:")
st.write(data.isnull().sum())

# Count the occurrences of each language
st.write("Language distribution:")
st.write(data["language"].value_counts())

# Train the machine learning model
x = data["Text"]
y = data["language"]
cv = sk.CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.33, random_state=42)

# Train the Multinomial Naive Bayes model
model = sk.MultinomialNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
st.write("Model accuracy:", score)

# User input
user_input = st.text_input("Enter a text:")

# Make predictions based on user input
if user_input:
    data = cv.transform([user_input]).toarray()
    output = model.predict(data)
    st.write("Predicted language:", output[0])
