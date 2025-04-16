# COMP 6600 AI Project 
# Project Computer Program
# Elizabeth Casey & Maddie Larkin

#import libraries
#pip install requests beautifulsoup4 scrapy pandas nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tweepy
import os
import time
from tweepy.errors import TooManyRequests

#Need to download only once
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')


#Collect Social Media Data
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALnt0QEAAAAAJ2I3yfphRhMkowzco00yHeCycoc%3DbyOHz6KF5vzndlyKdqJhPIxCK9NOgpWPhQQcPOHZtal7VjfdpL"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

query = '"Auburn University" OR "War Eagle" OR #AuburnUniversity OR #WarEagle OR "Auburn, AL", OR "Auburn Tigers" OR #AuburnTigers -is:retweet lang:en'

LAST_TWEET_ID = 'last_tweet_id.txt'
since_id = None
if os.path.exists(LAST_TWEET_ID):
    with open(LAST_TWEET_ID, 'r') as f:
        since_id = int(f.read().strip())
        since_id = int(since_id) if since_id else None


try:    
    tweets = client.search_recent_tweets(
        query=query, 
        max_results=100, 
        tweet_fields=['created_at', 'text', 'author_id'],
        since_id=since_id
        )

    tweet_data = []
    if tweets.data:
        for tweet in tweets.data:
            tweet_data.append({
                'id': tweet.id,
                'created_at': tweet.created_at,
                'text': tweet.text,
                'author_id': tweet.author_id
                })

        df = pd.DataFrame(tweet_data)
        df.to_csv('auburn_tweets.csv', mode = 'a', index=False, header= not os.path.exists('auburn_tweets.csv'))

        max_id = max(tweet.id for tweet in tweets.data)
        with open(LAST_TWEET_ID, 'w') as f:
            f.write(str(max_id))

        print(f'Collected {len(df)} new tweets')
    else:
        print('No new tweets found')

except TooManyRequests:
    #print("Rate limit exceeded. Please wait 15 minutes")
    #time.sleep(15 * 60)
    pass


#Data Preprocessing
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocessing_text(text):
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) #remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers

    tokens = nltk.word_tokenize(text)  # Tokenize the text

    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]  # Remove stopwords and lemmatize

    return ' '.join(cleaned)  # Join the cleaned tokens back into a string

df =  pd.read_csv('auburn_tweets.csv')
df['cleaned_text'] = df['text'].apply(preprocessing_text)
df.to_csv('auburn_tweets_cleaned.csv', index=False)
#print("Preprocessing complete.")



#Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
custom_words = {
    'war eagle': 4.0,
    'plains': 2.0
}
analyzer.lexicon.update(custom_words)
def sentiment_analysis(text):
    scores = analyzer.polarity_scores(text)['compound']
    if scores >= 0.05:
        sentiment = 'positive'
    elif scores <= -0.05:
        sentiment = 'negative'
    else:
        sentiment =  'neutral'
    return sentiment

df = pd.read_csv('auburn_tweets_cleaned.csv')
df['sentiment'] = df['cleaned_text'].apply(sentiment_analysis)
df.to_csv('auburn_tweets_sentiment.csv', index=False)

#print(df[['text', 'cleaned_text', 'sentiment']].head())


#Build Linear Regression Model to analyze the sentiment of the tweets
df = pd.read_csv('auburn_tweets_sentiment.csv')
label_encoder = LabelEncoder()
df['sentiment_label'] = label_encoder.fit_transform(df['sentiment'])

#vecorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text'])

#split data
y = df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

def main():
    api = tweepy.API(auth)










