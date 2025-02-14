import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
training_data = ["Your free trial is expiring soon", "Hi, how are you?", "Special offer, buy now", "Let's grab lunch tomorrow"] #Edit it as per your requirements
labels = [1, 0, 1, 0]  # 1: Spam, 0: Not Spam

# Step 1: Vectorize the data
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(training_data)

# Step 2: Train the model
model = MultinomialNB()
model.fit(X_train, labels)

# Step 3: Save the model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")
