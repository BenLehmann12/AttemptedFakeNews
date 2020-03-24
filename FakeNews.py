import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import plot_confusion_matrix


news = pd.read_csv("news.csv")
print(news)
X = news['text']
Y = news.label

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state=7)

count_vector = CountVectorizer(stop_words='english')
count_train = count_vector.fit_transform(x_train)
count_test = count_vector.transform(x_test)

tf_vector = TfidfVectorizer(stop_words='english')
tf_train = tf_vector.fit_transform(x_train)
tf_test = tf_vector.transform(x_test)

classifier = MultinomialNB()   #about 88.7% accurate
classifier.fit(count_train,y_train)
predict = classifier.predict(count_test)
score = accuracy_score(y_test, predict)
print(score)
cm = confusion_matrix(y_test, predict, labels=['FAKE','REAL'])
print(cm)

linear_class = PassiveAggressiveClassifier(max_iter=50)  #about 92.34% accurate
linear_class.fit(tf_train, y_train)
y_pred = linear_class.predict(tf_test)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)
c_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
print(c_matrix)

NB_classifier = MultinomialNB(alpha=1.0)
NB_classifier.fit(tf_train,y_train)
Group_labels = NB_classifier.classes_
feature_names = tf_vector.get_feature_names()
Weights = sorted(zip(NB_classifier.coef_[0], feature_names))
print(Group_labels[0], Weights[:20])
print(Group_labels[1], Weights[-20:])