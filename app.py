import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to transform the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    st.success("Model and vectorizer loaded successfully!")

except FileNotFoundError:
    st.error("Model or Vectorizer file not found! Please check the file paths.")
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer: {e}")

# Streamlit app layout
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Classify"):
    if input_sms.strip() == "":
        st.warning("Please enter a message!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize (Ensure input is an array)
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        try:
            result = model.predict(vector_input)[0]

            # 4. Display
            if result == 1:
                st.header("🚨 Spam 🚨")
            else:
                st.header("✅ Not Spam ✅")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
