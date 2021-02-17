import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

df = pd.read_csv("spam.csv", encoding="latin-1")

	
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
	
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
	
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

#Alternative Usage of Saved Model
joblib.dump(clf, 'spammail.pkl')
joblib.dump(cv, 'cv.pkl')
