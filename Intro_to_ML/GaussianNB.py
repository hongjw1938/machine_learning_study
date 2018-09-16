from sklearn.naive_bayse import GaussianNB
clf = GaussianNB()
X = []
Y = []
clf.fit(X, Y) # Lets suppose we have enough example that X, Y
# X is featues, Y is labels
test = []
test_labels = []
pred = clf.predict(test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test, test_labels)
# 위와 같이, 정확도를 구할 수 있다. 실제 label에 어느정도 fit하는지.