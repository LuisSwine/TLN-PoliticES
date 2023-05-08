import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.display import display

#Instanciamos el Spacy
nlp = spacy.load('es_core_news_sm')
spanish_stopwords = spacy.lang.es.stop_words.STOP_WORDS

df = pd.read_csv('df-corpus.csv')

grupos = df.groupby('Bi-Ideologia')

df_clase_0 = grupos.get_group(0)
df_clase_1 = grupos.get_group(1)

tweets = df['Tweets'].tolist()
tweets_0 = df_clase_0['Tweets'].tolist()
tweets_1 = df_clase_1['Tweets'].tolist()

vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, min_df=5, ngram_range=(1, 2))

tfidf_matrix = vectorizer.fit_transform(tweets)
tfidf_matrix_0 = vectorizer.fit_transform(tweets_0)
tfidf_matrix_1 = vectorizer.fit_transform(tweets_1)

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['Bi-Ideologia'], random_state = 0, test_size=0.1)
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(tfidf_matrix_0, df_clase_0['Multi-Ideologia'], random_state = 0, test_size=0.1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(tfidf_matrix_1, df_clase_1['Multi-Ideologia'], random_state = 0, test_size=0.1)

# Crear un modelo de regresión logística
clf = LogisticRegression()
clf_0 = LogisticRegression()
clf_1 = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)
clf_0.fit(X_train_0, y_train_0)
clf_1.fit(X_train_1, y_train_1)

scores = cross_val_score(clf, X_train, y_train, cv=5)
scores_0 = cross_val_score(clf_0, X_train_0, y_train_0, cv=5)
scores_1 = cross_val_score(clf_1, X_train_1, y_train_1, cv=5)

# Evaluar el rendimiento del modelo en los datos de prueba
score = clf.score(X_test, y_test)
score_0 = clf_0.score(X_test_0, y_test_0)
score_1 = clf_1.score(X_test_1, y_test_1)

mean_score = scores.mean()
mean_score_0 = scores_0.mean()
mean_score_1 = scores_1.mean()

y_pred = clf.predict(X_test)
y_pred_0 = clf_0.predict(X_test_0)
y_pred_1 = clf_1.predict(X_test_1)

accuracy = accuracy_score(y_test, y_pred)
accuracy_0 = accuracy_score(y_test_0, y_pred_0)
accuracy_1 = accuracy_score(y_test_1, y_pred_1)

print('Clasificacion left-right')
print(f"Score {score}")
print(f"CV Score {mean_score}")
print(f"Accuracy {accuracy}")

print('\nClasificacion sub-clases 0')
print(f"Score_0 {score_0}")
print(f"CV Score_0 {mean_score_0}")
print(f"Accuracy_0 {accuracy_0}")

print('\nClasificacion sub-clases 1')
print(f"Score_1 {score_1}")
print(f"CV Score_1 {mean_score_1}")
print(f"Accuracy_1 {accuracy_1}")