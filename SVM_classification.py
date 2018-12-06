from sklearn import svm
import numpy as np
import os, pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

features = np.load('featuredExtract.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

x_train = features[:30]
x_test = features[30:]

# clf = svm.SVC(gamma='auto')

clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)
pickle.dump(clf,open('modelSVM.pkl','wb'))

test_predict_Y= clf.predict(x_test)
print(test_predict_Y)
print('Accurancy: {}'.format(accuracy_score(y_test,test_predict_Y)))
# print(result)
# pickle.dump(result,open('result.pkl','wb'))
# clf = pickle.load(open('modelSVM.pkl','rb'))
