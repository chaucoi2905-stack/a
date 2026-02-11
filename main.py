import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Load dataset from CSV file
data = pd.read_csv('omicron.csv')
print(data.head())
print(data.isnull().sum())

#Import additional libraries for text processing
import nltk
nltk.download('vader_lexicon')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopwords = set(stopwords.words('english'))

#Define function to clean text data
def clean(text) :
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    text = ' '.join([stemmer.stem(word) for word in tokens if word not in stopwords])
    return text

#Apply text cleaning function to the 'text' column in dataset
data['cleaned_text'] = data['text'].apply(clean)

#Generate word cloud from cleaned text 
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
#Create word cloud
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("sentiment.png")
plt.close()

#Perform sentiment analysis using VADER
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

#Extract sentiment scores for each text entry
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["cleaned_text"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["cleaned_text"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["cleaned_text"]]

#Keep only relevant columns
data = data[["text", "Positive", "Negative", "Neutral"]]
print(data.head())

#Sum up sentiment scores across all text entries
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

#Define function to determine overall sentiment
def analyze_sentiment(a,b, c):
    if (a > b) and (a > c):
       print("Positive ?")
    elif (b > a) and (b > c):
         print("Negative ?")
    else:
            print("Neutral ?")

#Call function with aggregated sentiment scores
print("Sentiment analysis completed successfully.")




