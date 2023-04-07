import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train= pd.read_csv(r"C:/Users/moha0003/Desktop/Python/Competitions/Loan Prediction/train.csv")
test= pd.read_csv(r"C:/Users/moha0003/Desktop/Python/Competitions/Loan Prediction/test.csv")

print(train)
print(test)

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

train.head()
test.head()

train.describe()
train.isnull().sum()

train.drop("Loan_ID",axis=1,inplace=True)
test.drop("Loan_ID",axis=1,inplace=True)
test.head()

# Process for train data set
     
def Gender(Gender):
    if Gender == 'Male':
        return 1
    elif Gender == 'Female':
        return 2
    else:
        return 3

train['Gender'] = train['Gender'].apply(Gender)

def Married(Married):
    if Married == "Yes":
        return 1
    elif Married =="No":
        return 2
    else:
        return 3

train["Married"] = train["Married"].apply(Married)

def Dependents(Dependents):
    if Dependents =="0":
        return 0
    elif Dependents =="1":
        return 1
    elif Dependents =="2":
        return 2
    elif Dependents =="3+":
        return 3
    else:
        return 4
    
train["Dependents"]=train['Dependents'].apply(Dependents)


def Education(Education):
    if Education == "Graduate":
        return 1
    elif Education == "Not Graduate":
        return 2

train["Education"]=train["Education"].apply(Education)


def Self_Employed(Self_Employed):
    if Self_Employed =="Yes":
        return 1
    elif Self_Employed=="No":
        return 2
    else:
         return 3

train["Self_Employed"]=train["Self_Employed"].apply(Self_Employed)
    

def Property_Area(Property_Area):
    if Property_Area =="Urban":
        return 1
    elif Property_Area=="Rural":
        return 2
    else:
        return 3
    
train["Property_Area"]=train["Property_Area"].apply(Property_Area)


def Loan_Status(Loan_Status):
    if Loan_Status=="Y":
        return 1
    else:
        return 0

train["Loan_Status"]=train["Loan_Status"].apply(Loan_Status)

train["LoanAmount"]=train["LoanAmount"].fillna(train["LoanAmount"].mean())
train["Loan_Amount_Term"]=train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mean())
train=train.dropna(subset=["Credit_History"])

# Procedure for test dataset

def Gender(Gender):
    if Gender =='Male':
        return 1
    elif Gender=="Female":
        return 2
    else:
        return 3
test["Gender"]=test["Gender"].apply(Gender)

def Dependents(Dependents):
    if Dependents=="0":
        return 0
    elif Dependents =="1":
        return 1
    elif Dependents=="2":
        return 2
    elif Dependents=="3+":
        return 3
    else:
        return 4
    
test["Dependents"]=test["Dependents"].apply(Dependents)    
    
def Self_Employed(Self_Employed):
    if Self_Employed=="Yes":
        return 1
    elif Self_Employed=="No":
        return 2
    else:
        return 3

test["Self_Employed"]=test["Self_Employed"].apply(Self_Employed)

def Married(Married):
    if Married=="Yes":
        return 1
    else:
        return 2

test["Married"]=test["Married"].apply(Married)

def Education(Education):
    if Education == "Graduate":
        return 1
    elif Education == "Not Graduate":
        return 2
test["Education"]=test["Education"].apply(Education)

def Property_Area(Property_Area):
    if Property_Area =="Urban":
        return 1
    elif Property_Area=="Rural":
        return 2
    else:
        return 3
    
test["Property_Area"]=test["Property_Area"].apply(Property_Area)



test=test.dropna(subset=["Credit_History"])

test["LoanAmount"]=test["LoanAmount"].fillna(test["LoanAmount"].mean())
test["Loan_Amount_Term"]=test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].mean())


test.isnull().sum()

filepath="C:/Users/moha0003/Desktop/Python/Competitions/Loan Prediction/trail2.csv"

test.to_csv(filepath, index=False)

X = train.drop('Loan_Status',axis=1)
y = train['Loan_Status']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.1,random_state=101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test = sc.transform(test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train , Y_train)

pred = dtc.predict(X_test)
pred

from sklearn.metrics import accuracy_score, confusion_matrix
dtc_acc = accuracy_score(pred,Y_test)
print(dtc_acc)
print(confusion_matrix(pred,Y_test))

from sklearn.svm import SVC
knn=SVC()
knn.fit(X_train,Y_train)

pred1 = knn.predict(X_test)
pred1

from sklearn.metrics import accuracy_score
svc_acc = accuracy_score(pred1,Y_test)
print(svc_acc)
print(confusion_matrix(pred1,Y_test))

from sklearn.linear_model import LogisticRegression  # its a classification
lr=LogisticRegression()
lr.fit(X_train,Y_train)

pred2 = lr.predict(X_test)
pred2

from sklearn.metrics import accuracy_score
lr_acc = accuracy_score(pred2,Y_test)
print(lr_acc)
print(confusion_matrix(pred2,Y_test))

from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier()
gbm.fit(X_train,Y_train)

pred3= gbm.predict(X_test)

from sklearn.metrics import accuracy_score
gbm_acc = accuracy_score(pred2,Y_test)
print(gbm_acc)
print(confusion_matrix(pred2,Y_test))


plt.bar(x=['dtc','svc','lr','gbm'],height=[dtc_acc,svc_acc,lr_acc,gbm_acc])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()

filepath="C:/Users/moha0003/Desktop/Python/Competitions/Loan Prediction/predictions.csv"


predictions=lr.predict(test)
output = pd.DataFrame({'Predicted': predictions})
output.to_csv(filepath, index=False)
