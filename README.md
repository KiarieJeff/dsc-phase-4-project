## <span style="font-family:Georgia, serif;">**Twitter Sentiment Analysis** :Understanding Emotions in Tweets about Apple and Google products.</span>

![alt text](twits.jpg "Title")

## Overview

In this project, we will begin by constructing a straightforward baseline model for sentiment prediction, categorizing sentiments as either positive or negative. Subsequently, we will advance our approach by building a more sophisticated model capable of recognizing sentiments as positive, negative, or neutral. Our methodology will heavily rely on Natural Language Processing (NLP) techniques to preprocess the text data into a format suitable for model interpretation.

In the modeling phase, we will employ both supervised and unsupervised learning techniques. On the supervised front, we will create a pipeline that integrates multiple classifiers, enabling us to evaluate their performance through cross-validation scores. In the unsupervised domain, we will develop a neural network that functions similarly to the classifiers, utilizing it for sentiment prediction on textual data.

Leveraging our neural network, we will carry out sentiment predictions for a document and conduct evaluations specifically focused on product-related sentiments.

## Business Understanding

**Business Problem**: Using Sentiment Analysis to Improve Apple and Google Product Marketing Strategies 

The introduction of social media has completely changed how businesses interact with their consumers and the general public in today's connected society. While the digital age offers limitless possibilities for marketing and brand development, it also brings its own set of difficulties. One of these difficulties is the inability of enterprises to precisely gauge public opinion and feelings towards their goods or services.

In the age of social media, organizations are acutely aware of the need to harness the wealth of sentiment and emotion data available on these platforms. However, they often struggle to do so effectively, given the unprecedented speed, diversity, and complexity of social media communication. The dynamic nature of the medium, the diverse and contextual language used, the rapid increase of emojis and visual content, the volume of noise, and ethical concerns all contribute to the challenge of gauging public sentiment and emotions. 

To overcome these challenges, organizations must invest in advanced sentiment analysis tools and technologies, develop cultural and linguistic expertise, and strike a balance between data-driven insights and ethical considerations. By doing so, they can unlock the valuable insights hidden within the social media storm and use them to inform strategic decisions, enhance products and services, and build stronger connections with their audience in this rapidly evolving digital landscape.

## Data understanding

**Dataset Overview:**
This dataset comprises 8,721 entries organized into three distinct columns: tweet_text, emotion_in_tweet_is_directed_at, and is_there_an_emotion_directed_at_a_brand_or_product. Each entry in the dataset represents a tweet along with associated metadata regarding the product or brand it's directed at and the emotional sentiment conveyed within the tweet.

**Column Descriptions:**

`tweet_text:`

The tweet_text column contains the textual content of individual tweets. These tweets are typically concise, informal expressions shared by users on a social media platform, such as Twitter. Each tweet serves as a snapshot of a user's thoughts, opinions, or experiences related to a particular product or brand.
The textual data within this column can vary in length, language, and complexity. It may include hashtags, mentions of other users, URLs, and a wide range of linguistic elements.
Analysis of the tweet text can provide valuable insights into the sentiments, opinions, or feedback expressed by users regarding the product or brand.

`emotion_in_tweet_is_directed_at:`

The emotion_in_tweet_is_directed_at column provides information about the specific product or brand mentioned or targeted by each tweet. This column serves as a categorical label indicating the entity towards which the emotion or sentiment expressed in the tweet is directed.
Entries in this column may include the names or identifiers of various products or brands, allowing for the categorization of tweets based on the entity they reference.
Understanding which products or brands are most frequently mentioned in tweets can help identify consumer preferences and the areas where sentiment analysis may be most relevant.

`is_there_an_emotion_directed_at_a_brand_or_product:`

The is_there_an_emotion_directed_at_a_brand_or_product column characterizes the emotional sentiment or tone conveyed within each tweet directed at a product or brand.
This column serves as a crucial indicator of the emotional context of the tweets and can be categorized into several classes, including:

* Positive: Tweets expressing favorable sentiments, such as satisfaction, excitement, or endorsement, towards the product or brand.

* Negative: Tweets containing unfavorable sentiments, such as criticism, frustration, or dissatisfaction, directed at the product or brand.

* No Emotion: Tweets that do not convey any discernible emotional sentiment. These tweets may provide neutral or factual information.

* Not Clear: Tweets where the emotional tone is ambiguous or unclear, making it challenging to determine the sentiment.

Analyzing this column allows for sentiment classification and provides valuable insights into how consumers perceive and react to products or brands in the context of social media.
Dataset Size:

The dataset contains a total of 8,721 entries, each representing a unique tweet. This dataset size is substantial and provides a rich source of data for sentiment analysis and brand/product perception studies.

**Data Exploration and Analysis:**

To gain a deeper understanding of the dataset and its implications, exploratory data analysis (EDA) techniques, natural language processing (NLP) methods, and sentiment analysis tools can be applied.
EDA involves a series of techniques and methods to gain insights into the structure, content, and patterns within the textual information.

## Data Preparation

Text Preprocessing:Before conducting EDA, it's crucial to preprocess the text data. Common preprocessing steps include:
* Lowercasing: Convert all text to lowercase to ensure consistency.
* Tokenization: Split text into individual words or tokens.
* Stop Word Removal: Eliminate common and uninformative words like "the," "and," "in.
* Punctuation Removal: Remove special characters, punctuation marks, and symbols.
* Lemmatization or Stemming: Reduce words to their root form for better analysis.

Natural Language Processing (NLP) relies on regression techniques, necessitating the transformation of text data into numerical vectors that machine learning algorithms can comprehend and process effectively. This conversion can be achieved through various methods, such as GloVe (Global Vectors for Word Representation), Word2Vec, TF-IDF (Term Frequency-Inverse Document Frequency), and BERT (Bidirectional Encoder Representations from Transformers). In our particular scenario, we will opt for TF-IDF, which we ill be incorporating into a function designed to convert our text data. This function will also incorporate Compressed Sparse Row (CSR) matrices to transform the TF-IDF training matrix into a space-efficient CSR matrix format.

## Text Analysis

First we can calculate word frequencies, and print the top 10 most frequent words along with their normalized frequencies

```python
tweets_freqdist_top_10 = tweets_freqdist.most_common(10)
print(f'{"Word":<10} {"Normalized Frequency":<20}')
for word in tweets_freqdist_top_10:
    normalized_frequency = word[1] / total_word_count
    print(f'{word[0]:<10} {normalized_frequency:^20.4}')

```
| Word        | Normalized Frequency |
| ----------- | -----------          |
| sxsw        | 0.0858               |
| mention     | 0.06442              |
| link        | 0.03821              |
| rt          | 0.02743              |

     