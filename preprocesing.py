#IMPORTACIONES
import pandas as pd

import normalizacion
import lemma_token

from IPython.display import display
from sklearn import preprocessing

#Definimos la ruta del corpus
path_corpus = "politicES_phase_2_train_public.csv"

#Abrimos el corpus como un archivo de pandas
df_train = pd.read_csv(path_corpus)

#Ahora agrupamos con base en el id
df_train_grouped = df_train.groupby(["label", "gender", "profession", "ideology_binary", "ideology_multiclass"]).sum()

dict = {
        #'Autor': [trainDataset.iloc[0].name[0]],
        'Genero': [df_train_grouped.iloc[0].name[1]],
        'Profesion': [df_train_grouped.iloc[0].name[2]],
        'Tweets': df_train_grouped.iloc[0,0],
        'Bi-Ideologia': [df_train_grouped.iloc[0].name[3]],
        'Multi-Ideologia': [df_train_grouped.iloc[0].name[4]]
        }

df = pd.DataFrame(dict)

for i in range(1, len(df_train_grouped)):
    df.loc[i] = {
            #'Autor': trainDataset.iloc[i].name[0],
            'Genero': df_train_grouped.iloc[i].name[1],
            'Profesion': df_train_grouped.iloc[i].name[2],
            'Tweets': df_train_grouped.iloc[i,0],
            'Bi-Ideologia': df_train_grouped.iloc[i].name[3],
            'Multi-Ideologia': df_train_grouped.iloc[i].name[4]
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

df.to_csv('df-pre-corpus.csv', index=False)