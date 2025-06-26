import pandas as pd
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay

df=pd.read_csv("c:\\Users\\dubba\\Downloads\\email.csv")
df.rename(columns={"Category":"label","Message":"message"},inplace=True)
nltk.download('stopwords')
sw=set(stopwords.words('english'))
ps=PorterStemmer()
def clean(sent):
    sent=sent.lower()
    sent="".join([i for i in sent if i not in string.punctuation])
    words=sent.split()
    words=[i for i in words if i not in sw]
    words=[ps.stem(i) for i in words]
    return " ".join(words)
df["cleaned"]=df["message"].apply(clean)
df = df[df['label'].isin(['ham', 'spam'])].copy() #added here to remove the extra mode full thingy
vect=TfidfVectorizer(max_features=3000)
x=vect.fit_transform(df["cleaned"]).toarray()
le=LabelEncoder()
y=le.fit_transform(df["label"])
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=42)
nbmodel=MultinomialNB()
nbmodel.fit(x_train,y_train)
def predict_message(text, model, vectorizer):
    cleaned = clean(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "It is Spam" if prediction == 1 else "It is Ham"
y_train_pred=nbmodel.predict(x_train)
train_acc=accuracy_score(y_train,y_train_pred)
y_pred=nbmodel.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not spam","Spam"])
disp.plot()
plt.show()
test_acc=accuracy_score(y_test,y_pred)
print("Classification report for training data set : ",classification_report(y_train,y_train_pred))
print("Classification report for testing data set : ",classification_report(y_test,y_pred))
#print(df['label'].unique())
print(predict_message("Congratulations!!! You won an ipod on the lucky draw, claim the price by clicking on the link under the email", nbmodel, vect)) 


