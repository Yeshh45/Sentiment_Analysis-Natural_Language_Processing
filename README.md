!pip install transformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

df = pd.read_csv('/content/twitterds.csv').head(1000)

df = df[~((df['replies_count'] == 0) & (df['retweets_count'] == 0) & (df['likes_count'] == 0))]
df = df[df['language'] == 'en']

df = df[['tweet', 'replies_count', 'retweets_count', 'likes_count', 'time']].copy()
df['id'] = range(1, len(df) + 1)
df = df[['id', 'time', 'tweet', 'replies_count', 'retweets_count', 'likes_count']]
df.set_index('id', inplace=True)
df['total_engagement'] = df['replies_count'] + df['retweets_count'] + df['likes_count']
df.head()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words
df['processed_tweet'] = df['tweet'].apply(preprocess_text)
all_keywords = set()
for words_list in df['processed_tweet']:
    all_keywords.update(words_list)

Model_Name = f"cardiffnlp/twitter-roberta-base-sentiment"
auto_tokenizer_ = AutoTokenizer.from_pretrained(Model_Name)
model = AutoModelForSequenceClassification.from_pretrained(Model_Name)

def polarity(example):
    ET = auto_tokenizer_(example, return_tensors='pt')
    result = model(**ET)
    obtained_scores = result[0][0].detach().numpy()
    obtained_scores = softmax(obtained_scores)
    final_result = {
        'negative' : obtained_scores[0],
        'neutral' : obtained_scores[1],
        'positive' : obtained_scores[2]
    }
    return final_result

res = {'negative':[], 'neutral':[], 'positive':[]}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['tweet']
    for key, value in polarity(text).items():
      res[key].append(value)

df_res = pd.DataFrame(res)
df = pd.concat([df, df_res], axis=1) 

def classify_sentiment(row):
    if row['positive'] > row['negative'] and row['positive'] > row['neutral']:
        return 'positive'
    elif row['negative'] > row['positive'] and row['negative'] > row['neutral']:
        return 'negative'
    else:
        return 'neutral'

df['classified_sentiment'] = df.apply(classify_sentiment, axis=1)

# Distribuions

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

df['positive'].plot(kind='hist', bins=20, ax=axes[0], title='Positive')
axes[0].spines[['top', 'right']].set_visible(False)

df['neutral'].plot(kind='hist', bins=20, ax=axes[1], title='Neutral')
axes[1].spines[['top', 'right']].set_visible(False)

df['negative'].plot(kind='hist', bins=20, ax=axes[2], title='Negative')
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# Show the combined image
plt.show()

# Create a bar plot for the count of tweets about each country
sns.countplot(x='tweet_about', data=df)
plt.title('Tweet Count by Country')
plt.xlabel('Country')  # Add x-axis label for better clarity
plt.ylabel('Tweet Count')  # Add y-axis label for better clarity
plt.show()

sentiment_cols = ['negative', 'neutral', 'positive']

# Box plot of sentiment scores
df[sentiment_cols].boxplot()
plt.title('Box Plot of Sentiment Scores')
plt.ylabel('Sentiment Score')  # Add y-axis label for better clarity
plt.show()

# Pair plot for numerical columns
sns.pairplot(df[['replies_count', 'retweets_count', 'likes_count'] + sentiment_cols])
plt.suptitle('Pair Plot of Numerical Columns', y=1.02)
plt.show()

# Line plot of sentiment probabilities over time (assuming you have a datetime column)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df[sentiment_cols].plot(style='-')
plt.title('Sentiment Trends Over Time')
plt.xlabel('time')  # Add x-axis label for better clarity
plt.ylabel('Sentiment Probability')  # Add y-axis label for better clarity
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Scatter plot between total engagement and positive sentiment
sns.scatterplot(x='total_engagement', y='positive', data=df, hue='positive', palette='coolwarm', edgecolor='w', s=100, ax=axes[0])
axes[0].set_title('Total Engagement vs Positive Sentiment')
axes[0].set_xlabel('Total Engagement')
axes[0].set_ylabel('Positive Sentiment')
axes[0].legend(title='Positive Sentiment')

# Scatter plot between total engagement and neutral sentiment
sns.scatterplot(x='total_engagement', y='neutral', data=df, hue='neutral', palette='coolwarm', edgecolor='w', s=100, ax=axes[1])
axes[1].set_title('Total Engagement vs Neutral Sentiment')
axes[1].set_xlabel('Total Engagement')
axes[1].set_ylabel('Neutral Sentiment')
axes[1].legend(title='Neutral Sentiment')

# Scatter plot between total engagement and negative sentiment
sns.scatterplot(x='total_engagement', y='negative', data=df, hue='negative', palette='coolwarm', edgecolor='w', s=100, ax=axes[2])
axes[2].set_title('Total Engagement vs Negative Sentiment')
axes[2].set_xlabel('Total Engagement')
axes[2].set_ylabel('Negative Sentiment')
axes[2].legend(title='Negative Sentiment')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined image
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='tweet_about', hue='classified_sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution for Different Topics')
plt.xlabel('Tweet About')
plt.ylabel('Count')
plt.show()

df_grouped = df.groupby(['tweet_about', 'classified_sentiment']).size().unstack(fill_value=0)
df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0)

# Plot stacked bar plot
colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
ax = df_grouped.plot(kind='bar', stacked=True, color=[colors[col] for col in df_grouped.columns])

# Customize plot
plt.title('Sentiment Distribution for Different Topics (Scaled)')
plt.xlabel('Tweet About')
plt.ylabel('Proportion')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
