import re
import string
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

with open('logprior1.pkl','rb') as f:
    logPrior, logLikelihood, vocab = pickle.load(f)

with open('freqs.pkl', 'rb') as f:
 freqs= pickle.load(f)

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):

    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def sigmoid(z):
  h = 1/(1+np.exp(-z))
  return h

def extract_features(tweet,freqs):
  word_l = process_tweet(tweet)
  x = np.zeros((1,3))
  x[0,0] = 1
  for word in word_l:
    x[0,1] += freqs.get((word,1.0),0)
    x[0,2] += freqs.get((word,0.0),0)
  assert(x.shape == (1,3))
  return x

def predict_tweet(tweet, freqs, theta):
  x = extract_features(tweet,freqs)
  y_pred = sigmoid(np.dot(x,theta))
  return y_pred

def testNaiveBayes(test_x, logPrior, logLikelihood, vocab = vocab, freqs = freqs, test_y = np.array([0, 1])):
  C = np.unique(test_y).astype(int)
  sum = list(logPrior.values())
  for c in C:
    tmpSum = 0
    for word in process_tweet(test_x):
      if not word in vocab:
        continue
      sum[c] += logLikelihood.get((word, c), 0)

  return np.argmax(sum)