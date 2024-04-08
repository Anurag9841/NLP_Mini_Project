
from utils import process_tweet,build_freqs,sigmoid,extract_features,predict_tweet,testNaiveBayes
import pickle
import streamlit as st
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
with open('J_theta.pkl', 'rb') as f:
  J, theta = pickle.load(f)

with open('freqs.pkl', 'rb') as f:
 freqs= pickle.load(f)

with open('logprior1.pkl','rb') as f:
    logPrior, logLikelihood, vocab = pickle.load(f)

with open('svm_model.pickle','rb') as f:
    model = pickle.load(f)

with open('random_forests1.pkl','rb') as f:
    model_rf = pickle.load(f)

def predictions_logistic(text):
    prediction = predict_tweet(text, freqs, theta)
    if prediction > 0.5:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"

def predictions_NB(text):
    prediction = testNaiveBayes(text, logPrior, logLikelihood)
    if prediction == 1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"
def predictions_SVM(text):
    X_test = np.array([extract_features(text, freqs)])
    X_test = X_test.reshape(X_test.shape[0], -1)
    prediction = model.predict(X_test)
    if prediction == 1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"

def predictions_RF(tweet, _freqs = freqs, _classifier = model_rf):
  x = extract_features(tweet,_freqs)
  y_pred =_classifier.predict(x)
  if y_pred == 1:
      return "Positive Sentiment"
  else:
      return "Negative Sentiment"

def main():
    st.markdown('### Sentiment Analysis')
    input_string = st.text_input('Input Sentence to Test')

    if st.button('Predict Sentiment'):
        st.markdown('### Logistic Regression')
        sentiment_logistic = predictions_logistic(input_string)
        if sentiment_logistic == "Positive Sentiment":
            st.success("Sentiment (Logistic Regression): Positive 😊")
        else:
            st.error("Sentiment (Logistic Regression): Negative 😞")

        st.markdown('### Naive Bayes')
        sentiment_NB = predictions_NB(input_string)
        if sentiment_NB == "Positive Sentiment":
            st.success("Sentiment (Naive Bayes): Positive 😊")
        else:
            st.error("Sentiment (Naive Bayes): Negative 😞")

        st.markdown('### Support Vector Machine')
        sentiment_SVM = predictions_SVM(input_string)
        if sentiment_SVM == "Positive Sentiment":
            st.success("Sentiment (Support Vector Machine): Positive 😊")
        else:
            st.error("Sentiment (Support Vector Machine): Negative 😞")

        st.markdown('### Random Forest Classifier')
        sentiment_RF = predictions_RF(input_string)
        if sentiment_RF == "Positive Sentiment":
            st.success("Sentiment (Support Vector Machine): Positive 😊")
        else:
            st.error("Sentiment (Support Vector Machine): Negative 😞")

if __name__ == '__main__':
    main()



