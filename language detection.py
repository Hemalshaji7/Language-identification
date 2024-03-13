#importing python libraries
import pandas as pd 
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


#extrating the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
data.head(10).to_csv('output.txt', sep='\t', index=False)
print(data.isnull().sum())
data["language"].value_counts()

#trainig the machine language model using multinomial naiive Bayes
x=np.array(data["Text"])
y=np.array(data["language"])
cv=CountVectorizer()
X=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

#algorithm based on multinomial naiive Bayes to train language detection model
model=MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

#'taking the user input'
user=input("eneter a text: ")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)