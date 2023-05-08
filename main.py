#Realizaos las importaciones
import pandas as pd
from IPython.display import display
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score

import normalizacion
import lemma_token

def normalize(corpus):
    sample = []
    temp1=""
    for j in range(len(corpus)):
        temp1+= corpus[j] + " "
    sample.append(temp1)
    return sample


#Primero definimos las rutas donde tenemos toda la informacion
train_path = "politicES_phase_2_train_public.csv"
test_path = "politicES_phase_2_train_public.csv"

#Abrimos los dataset y realizamos la agrupacion con base en la label
trainDataset = pd.read_csv(train_path)
testDataset = pd.read_csv(test_path)

trainDataset = trainDataset.groupby(["label", "gender", "profession", "ideology_binary", "ideology_multiclass"]).sum()
testDataset = testDataset.groupby(["label", "gender", "profession", "ideology_binary", "ideology_multiclass"]).sum()

#display(trainDataset)
#display(testDataset)

dict = {
        #'Autor': [trainDataset.iloc[0].name[0]],
        'Genero': [trainDataset.iloc[0].name[1]],
        'Profesion': [trainDataset.iloc[0].name[2]],
        'Tweets': trainDataset.iloc[0,0],
        'Bi-Ideologia': [trainDataset.iloc[0].name[3]],
        'Multi-Ideologia': [trainDataset.iloc[0].name[4]]
        }

df = pd.DataFrame(dict)

for i in range(1, len(trainDataset)):
    df.loc[i] = {
            #'Autor': trainDataset.iloc[i].name[0],
            'Genero': trainDataset.iloc[i].name[1],
            'Profesion': trainDataset.iloc[i].name[2],
            'Tweets': trainDataset.iloc[i,0],
            'Bi-Ideologia': trainDataset.iloc[i].name[3],
            'Multi-Ideologia': trainDataset.iloc[i].name[4]
            }

for i in range(len(testDataset)):
    df.loc[i+len(trainDataset)] = {
            #'Autor': testDataset.iloc[i].name[0],
            'Genero': testDataset.iloc[i].name[1],
            'Profesion': testDataset.iloc[i].name[2],
            'Tweets': testDataset.iloc[i,0],
            'Bi-Ideologia': testDataset.iloc[i].name[3],
            'Multi-Ideologia': testDataset.iloc[i].name[4]
            }

encoder = preprocessing.LabelEncoder()
encoder.fit(['male','female'])
df['Genero'] = encoder.transform(df['Genero'])

encoder.fit(pd.unique(df['Profesion']))
df['Profesion'] = encoder.transform(df['Profesion'])

encoder.fit(pd.unique(df['Bi-Ideologia']))
df['Bi-Ideologia'] = encoder.transform(df['Bi-Ideologia'])

encoder.fit(pd.unique(df['Multi-Ideologia']))
df['Multi-Ideologia'] = encoder.transform(df['Multi-Ideologia'])

tweets_normalizados = df['Tweets'].apply(normalizacion.normalizar_tweet)
df['Tweets'] = tweets_normalizados

corpus = []
for i in range(len(df)):
    print(f"Lematizando {i}")
    corpus.append(
        lemma_token.clean_text(df['Tweets'].loc[i])
    )       
df['Tweets'] = corpus


df.to_csv('df-corpus.csv', index=False)

data = corpus
data

corpusOfTweets=[]
for cleanTweet in corpus :
    corpusOfTweets.append(normalize(cleanTweet))
corpusOfTweets[0]

data_frame = pd.DataFrame()
data_frame[0] = [item for sublist in corpusOfTweets for item in sublist]

df_train, df_test = train_test_split(data_frame,test_size=0.2,random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(data_frame[0], df['Bi-Ideologia'], random_state = 0, test_size=0.2)

vectorizer1Gram = TfidfVectorizer(ngram_range=(1,1))
vectorizer2Gram = TfidfVectorizer(ngram_range=(1,2))


def tfidf_ngram(n_gram,X_train=X_train,X_test=X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1,n_gram))
    x_train_vec = vectorizer.fit_transform(X_train)
    x_test_vec = vectorizer.transform(X_test)
    return x_train_vec,x_test_vec

X_train1g_cv, X_test1g_cv = tfidf_ngram(1,X_train=X_train,X_test=X_test)
X_train2g_cv, X_test2g_cv = tfidf_ngram(2,X_train=X_train,X_test=X_test)


print(X_train1g_cv)
print(X_train2g_cv)

text_embedding = {
    'TF_IDF 1_gram':(X_train1g_cv,X_test1g_cv),
    'TF_IDF 2_gram':(X_train2g_cv,X_test2g_cv)
}

models = [
          LinearRegression(),
          BernoulliNB(),
          GaussianNB(),
          KNeighborsClassifier(),
          LogisticRegression()
          ]

results_dict={'Model Name':[],'Embedding type':[],'Testing Accuracy':[],'Cross Validation':[]}

for model in models:
    for embedding_vector in text_embedding.keys():
        train = text_embedding[embedding_vector][0].toarray()
        test = text_embedding[embedding_vector][1].toarray()
        model.fit(train, y_train)

        results_dict['Model Name'].append(type(model).__name__)
        results_dict['Embedding type'].append(embedding_vector)

        test_acc = model.score(test, y_test)
        results_dict['Testing Accuracy'].append(test_acc)

        score = cross_val_score(model,test,y_test, scoring='r2')
        results_dict['Cross Validation'].append(score.mean())

results_df=pd.DataFrame(results_dict)

results_df