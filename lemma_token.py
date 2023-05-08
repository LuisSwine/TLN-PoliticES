import re

import nltk
nltk.download('punkt')
nltk.download('stopwords') #'spanish'
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def get_wordnet_pos(word):
    # Mapeamos la etiqueta POS al primer caracter lemmatize() acepte
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    # Definicion de patron de palabras para la mantencion de estas y pasado de texto a minusculas
    lemmatizer = WordNetLemmatizer()
    sub_pattern = r'[^A-Za-z\u00C0-\u017F]+$]'
    split_pattern = r"\s+"
    stop_words = stopwords.words('spanish') + ['d', 'q', 'k', 'h']
    lower = text.lower()

    # Reemplazando todos los caracteres, excepto los que esten en los patrones definidos en sub_patten
    # a espacios, tokenizado de los documentos y lematizacion
    filtered = re.sub(sub_pattern,' ',lower).lstrip().rstrip()
    filtered = word_tokenize(filtered)
    filtered = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered if word not in stop_words]

    return filtered