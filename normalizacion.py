import re

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticonos
        u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # transporte y símbolos de mapa
        u"\U0001F1E0-\U0001F1FF"  # banderas (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalizar_tweet(tweet):
    # Convierte el tweet a minúsculas
    tweet = tweet.lower()
    
    # Remueve los emojis
    tweet = remove_emojis(tweet)
    
    # Remueve las menciones y hashtags
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    
    # Remueve los enlaces
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remueve los caracteres especiales y los números
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    
    # Remueve los espacios adicionales
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = tweet.strip()
    
    return tweet