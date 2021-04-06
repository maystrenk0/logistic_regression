from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

digits = load_digits()

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.get_cmap('gray'))
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
res = {}


def do_lin_reg(solver):
    lr = LogisticRegression(solver=solver)
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    score = lr.score(x_test, y_test)
    return score, predictions


for s in solvers:
    ans = do_lin_reg(s)
    # print(s)
    # print("Accuracy Score: ", ans[0])
    # print("F1 Score: ", f1_score(y_test, ans[1], average='macro'), '\n')
    res[s] = ans[0]

best_solver = max(res, key=res.get)
best_ans = do_lin_reg(best_solver)
print("Best solver: ", best_solver)
print("Accuracy Score: ", best_ans[0])
print("F1 Score: ", f1_score(y_test, best_ans[1], average='macro'), '\n')

cm = metrics.confusion_matrix(y_test, best_ans[1])
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(best_ans[0]), size=15)
plt.show()
