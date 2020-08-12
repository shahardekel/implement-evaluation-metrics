import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#A
def adjust_labels_to_binary (y_train, target_class_value):
    types = {'Setosa': 0, 'Versicolour': 1, 'Virgincacv': 2}
    result=np.where(y_train == types[target_class_value], 1, -1)
    return result

#B
def one_vs_rest( x_train, y_train, target_class_value):
    y_train_binarized=adjust_labels_to_binary(y_train,target_class_value)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train_binarized)
    return logreg

#C
def binarized_confusion_matrix (X, y_binarized, one_vs_rest_model, prob_threshold):
    TP, FP, FN, TN = 0, 0, 0, 0
    y_pred_prob = one_vs_rest_model.predict_proba(X)[:,1]
    y_binarized_pred = np.full_like(y_binarized,0)
    for i in range(len(y_pred_prob)):
        if y_pred_prob[i] >= prob_threshold:
            y_binarized_pred[i] = 1
        else:
            y_binarized_pred[i] = -1

        if y_binarized[i] == 1 and y_binarized_pred[i] == 1:
            TP += 1
        if y_binarized[i] == -1 and y_binarized_pred[i] == 1:
            FP += 1
        if y_binarized[i] == 1 and y_binarized_pred[i] == -1:
            FN += 1
        if y_binarized[i] == -1 and y_binarized_pred[i] == -1:
            TN += 1
    con_mat = [[TP, FP], [FN, TN]]
    return con_mat

#D
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=98)
types=['Setosa', 'Versicolour', 'Virgincacv']
for flower in types:
    model=one_vs_rest(X_train, y_train, flower)
    y_binarized_train=adjust_labels_to_binary(y_train,flower)
    train_cm=binarized_confusion_matrix(X_train, y_binarized_train, model, 0.5)
    y_binarized_test=adjust_labels_to_binary(y_test,flower)
    test_cm=binarized_confusion_matrix(X_test, y_binarized_test, model, 0.5)

    print("The confusion matrix for the train data of",flower,"is:",train_cm)
    print("The confusion matrix for the test data of", flower, "is:", test_cm)
    print()

#E
def micro_avg_precision(X, y, all_targed_class_dict, prob_threshold):
    TP,FP=0,0
    types = ['Setosa', 'Versicolour', 'Virgincacv']
    for flower in types:
        y_test_binary=adjust_labels_to_binary(y,flower)
        con_mat=binarized_confusion_matrix(X,y_test_binary,all_targed_class_dict[flower],prob_threshold)
        TP+= con_mat[0][0]
        FP+=con_mat[0][1]

    res=TP/(TP+FP)
    return res

#F
def micro_avg_recall(X, y, all_targed_class_dict, prob_threshold):
    TP, FN = 0,0
    types = ['Setosa', 'Versicolour', 'Virgincacv']
    for flower in types:
        y_test_binary = adjust_labels_to_binary(y, flower)
        con_mat = binarized_confusion_matrix(X, y_test_binary, all_targed_class_dict[flower], prob_threshold)
        TP += con_mat[0][0]
        FN += con_mat[1][0]

    res = TP / (TP + FN)
    return res

#G
def micro_avg_false_positive_rate(X, y, all_targed_class_dict, prob_threshold):
    FP, TN = 0,0
    types = ['Setosa', 'Versicolour', 'Virgincacv']
    for flower in types:
        y_test_binary = adjust_labels_to_binary(y, flower)
        con_mat = binarized_confusion_matrix(X, y_test_binary, all_targed_class_dict[flower], prob_threshold)
        FP += con_mat[0][1]
        TN += con_mat[1][1]

    res = FP / (FP + TN)
    return res

#H
thresholds= [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]
fpr=[]
recall_H=[]
types = ['Setosa', 'Versicolour', 'Virgincacv']
all_targed_class_dict = {}

for flower in types:
    all_targed_class_dict[flower]=one_vs_rest(X_train,y_train,flower)

for i in thresholds:
    fpr.append(micro_avg_false_positive_rate(X_test, y_test, all_targed_class_dict, i))
    recall_H.append(micro_avg_recall(X_test, y_test, all_targed_class_dict, i))

plt.scatter(fpr, recall_H)
plt.plot(fpr, recall_H)
#plt.plot([0, 1], [0, 1],'r--')
plt.title("ROC curve for the test set")
plt.xlabel("False Positive Rate")
plt.ylabel("Average Recall Rate")
plt.grid(True)
plt.show()

#I
def f_beta(precision, recall, beta):
    return ((1+np.power(beta,2))*(precision*recall))/(np.power(beta,2)*precision+recall)

#J
f_beta_dict={0.3:[],0.5:[],0.7:[]}
threshold=[0.3,0.5,0.7]
for i in threshold:
    precision=micro_avg_precision(X_test, y_test, all_targed_class_dict, i)
    recall_J=micro_avg_recall(X_test, y_test, all_targed_class_dict, i)
    for beta in range(11):
        f_beta_dict[i] += [f_beta(precision, recall_J, beta)]

beta_x=range(11)
for i, color in zip([0.3, 0.5, 0.7], ['green', 'orange', 'red']):
    plt.plot(beta_x, f_beta_dict[i], color=color)

plt.ylabel('f_beta values')
plt.xlabel('beta')
plt.legend(['f_beta 0.3', 'f_beta 0.5', 'f_beta 0.7'])
plt.title('f_beta as a function of beta for test data')
plt.show()






