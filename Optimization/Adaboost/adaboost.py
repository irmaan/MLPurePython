
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def I(flag):
    if flag==1:
      return 1
    return 0  

def sign(x):
    
    if x!=0:
      return abs(x)/x 
    return 1       


class AdaBoost:
    def __init__(self, n_estimators=55):
        self.n_estimators=n_estimators
        self.models=[None]*n_estimators
        
    def fit(self,x,y):
        x=np.float64(x)
        Nu=len(y)
        w=np.array([1/Nu for i in range(Nu)])  
        for n in range(self.n_estimators):
            Gmm=DecisionTreeClassifier(max_depth=1)\
                 .fit(x,y,sample_weight=w).predict
                        
            error=sum([w[i]*I(y[i]!=Gmm(x[i].reshape(1,-1)))\
            for i in range(Nu)])/sum(w)

            Alpha=np.log((1-error)/error)
            
            w=[w[i]*np.exp(Alpha*I(y[i]!=Gmm(x[i].reshape(1,-1))))\
                        for i in range(Nu)]   
            
            self.models[n] = (Alpha,Gmm)

    def predict(self,x):
        y=0
        for n in range(self.n_estimators):
            Alpha, Gmm = self.models[n]
            y+=Alpha*Gmm(x)
        sign_=np.vectorize(sign)
        y=np.where(sign_(y)==-1,-1,1)
        return y

# load data

import pandas as pd
import numpy as np
df=pd.read_csv('Iris.csv')

y=df.Species
x=df.drop(['Id', 'Species'],axis=1)

# Main Algorithm 

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score 
x,y=make_classification(n_samples=150)

y=np.where(y==0,-1,1)     #as y needs to be in {-1,1} for our implementaion of AdaBoost Classifer
classification=AdaBoost(n_estimators=5)
classification.fit(x,y)
y_pred_classification=classification.predict(x)

print("Accuracy of the AdaBoost-Classifier Model (algorithm from scratch):",accuracy_score(y, y_pred_classification))

# Test

from sklearn.ensemble import AdaBoostClassifier

classification_check=AdaBoostClassifier(n_estimators=5, algorithm="SAMME")
classification_check.fit(x,y)
y_pred_classification_check=classification_check.predict(x)

print("Accuracy Check of the AdaBoost-Classifier Model (sklearn implementation):",accuracy_score(y, y_pred_classification_check))
