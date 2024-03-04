import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("processed.cleveland.data")

train = df.sample(frac=0.7, random_state=1)
test = df.drop(train.index)

y_train = train["num"]
x_train = train.drop("num", axis=1)

y_test = test["num"]
x_test = test.drop("num", axis=1)

clf = GaussianNB()
clf.fit(x_train, y_train)

print('Accuracy for train set:', round(clf.score(x_train, y_train), 5))
print('Accuracy for test set:', round(clf.score(x_test, y_test), 5))
