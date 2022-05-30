from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier,  VotingClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

main = tkinter.Tk()
main.title("Ensemble Model For Credit Scoring")
main.geometry("1300x1200")

global filename
global cols
global x,y
global X_train, X_test, y_train, y_test
global boost,tree_acc,svm_acc,random_acc,linear_acc,backflow_acc

def prediction(X_test, cls):
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test,y_pred)*100
    textarea.insert(END,details+"\n\n")
    textarea.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    textarea.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    return accuracy            



def upload():
    textarea.delete('1.0', END)
    global filename
    global cols
    global X,Y
    global X_train, X_test, y_train, y_test
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)    

    train = pd.read_csv(filename)
    le = preprocessing.LabelEncoder()
    train['Gender'] = le.fit_transform(train['Gender'])
    train['Married'] = le.fit_transform(train['Married'])
    train['Bank_Customer'] = le.fit_transform(train['Bank_Customer'])
    train['Education'] = le.fit_transform(train['Education'])
    train['Ethnicity'] = le.fit_transform(train['Ethnicity'])
    train['Years_Employed'] = le.fit_transform(train['Years_Employed'])
    train['Prior_Default'] = le.fit_transform(train['Prior_Default'])
    train['Credit_Score'] = le.fit_transform(train['Credit_Score'])
    train['Drivers_License'] = le.fit_transform(train['Drivers_License'])
    train['Approved'] = le.fit_transform(train['Approved'])

    
    

    cols = train.shape[1]
    X = train.values[:, 0:cols-1] 
    Y = train.values[:, cols-1]
    Y = Y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    textarea.insert(END,"Credit Score Train & Test Model Generated\n\n")
    textarea.insert(END,"Total Dataset Size : "+str(len(X))+"\n")
    textarea.insert(END,"Splitted Training Size : "+str(len(X_train))+"\n")
    textarea.insert(END,"Splitted Test Size : "+str(len(X_test))+"\n\n")

    textarea.insert(END,"Before Outlier Detection, counts of label '1': {}".format(sum(y_train == 1))+"\n") 
    textarea.insert(END,"Before Outlier Detection, counts of label '0': {} \n".format(sum(y_train == 0))+"\n")

    sm = SMOTE(random_state = 2) 
    X_train, y_train = sm.fit_resample(X_train, y_train)
    textarea.insert(END,"After SMOTE Outlier Detection, counts of label '1': {}".format(sum(y_train == 1))+"\n") 
    textarea.insert(END,"After SMOTE Outlier Detection, counts of label '0': {}".format(sum(y_train == 0))+"\n")


def boosting():
    textarea.delete('1.0', END)
    global boost
    cls = xgb.XGBClassifier(n_estimators=1, max_depth=1)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    boost = cal_accuracy(y_test, prediction_data,'Extreme Gradient Boosting Accuracy & Classification Details')
    

def tree():
    global tree_acc
    textarea.delete('1.0', END)
    cls = GradientBoostingClassifier(n_estimators=1, max_depth=1, random_state=0)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    tree_acc = cal_accuracy(y_test, prediction_data,'Gradient Boosting Decision Tree Accuracy & Classification Details')
    
def SVM():
    global svm_acc
    textarea.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 0)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy & Classification Details')
   
def randomForest():
    global random_acc
    textarea.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=1,random_state=0)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Accuracy & Classification Details')

def linear():
    global linear_acc
    textarea.delete('1.0', END)
    cls = LinearDiscriminantAnalysis()
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls) 
    linear_acc = cal_accuracy(y_test, prediction_data,'Linear Discriminant Analysis Accuracy & Classification Details')

def backFlow():
    global backflow_acc
    textarea.delete('1.0', END)
    cls1 = xgb.XGBClassifier(n_estimators=100, max_depth=100, learning_rate=0.01, subsample=0.01)
    cls2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=50, random_state=0)
    cls3 = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 0)
    cls4 = RandomForestClassifier(n_estimators=100,max_depth=50,random_state=0)
    cls5 = LinearDiscriminantAnalysis()
    cls6 = VotingClassifier(estimators=[
         ('xgb', cls1), ('dt', cls2), ('svm', cls3), ('rf', cls4), ('lda',cls5)], voting='hard')
    cls6.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls6) 
    backflow_acc = cal_accuracy(y_test, prediction_data,'Backflow Learning Accuracy & Classification Details')

def graph():
    height =[ boost,tree_acc,svm_acc,random_acc,linear_acc,backflow_acc]
    bars = ('XGB Accuracy','GB Decision Tree Acc','SVM Acc','Random Forest Acc','Linear Acc','Backflow Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos,height)
    plt.xticks(y_pos,bars) 
    plt.show()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='A Novel Noise-Adapted Two-Layer Ensemble Model For Credit Scoring Based On Backflow Learning')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Credit Dataset & Apply IFNA Processing", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='mint cream', fg='olive drab')  
pathlabel.config(font=font1)           
pathlabel.place(x=500,y=100)

boostingButton = Button(main, text="Run Extreme Gradient Boosting", command=boosting)
boostingButton.place(x=50,y=150)
boostingButton.config(font=font1) 

treeButton = Button(main, text="Run Gradient Boosting Decision Tree", command=tree)
treeButton.place(x=380,y=150)
treeButton.config(font=font1) 

svmButton = Button(main, text="Run SVM", command=SVM)
svmButton.place(x=750,y=150)
svmButton.config(font=font1)

randomButton = Button(main, text="Run Random Forest", command=randomForest)
randomButton.place(x=50,y=200)
randomButton.config(font=font1) 

linearButton = Button(main, text="Run Linear Discriminant Analysis", command=linear)
linearButton.place(x=280,y=200)
linearButton.config(font=font1)

backButton = Button(main, text="Run Backflow Learning", command=backFlow)
backButton.place(x=610,y=200)
backButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=880,y=200)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=20,width=150)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=250)
textarea.config(font=font1)


main.config(bg='gainsboro')
main.mainloop()
