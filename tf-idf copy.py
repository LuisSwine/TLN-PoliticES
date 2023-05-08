import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.display import display

import spacy
nlp = spacy.load('es_core_news_sm')
spanish_stopwords = spacy.lang.es.stop_words.STOP_WORDS

df = pd.read_csv('df-corpus.csv')
tweets = df['Tweets'].tolist()

vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, min_df=5)

tfidf_matrix = vectorizer.fit_transform(tweets)
    
df_indepen = pd.concat([df['Genero'], df['Profesion'], pd.DataFrame(tfidf_matrix.toarray())], axis=1)

#display(df_indepen)

X_train, X_test, y_train, y_test = train_test_split(df_indepen, df['Bi-Ideologia'], random_state = 0, test_size=0.2)

# Crear un modelo de regresión logística
clf = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)

# Evaluar el rendimiento del modelo en los datos de prueba
score = clf.score(X_test, y_test)

mean_score = scores.mean()

y_pred = clf.predict(X_test)

accuracy =accuracy_score(y_test, y_pred)

print(f"Score {score}")
print(f"CV Score {mean_score}")
print(f"Accuracy {accuracy}")