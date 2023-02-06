import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from IPython.display import display, HTML
import numpy as np

#import the data set
mydata = mypd.read_csv("/home/nasc/Documents/G/ML/doc/Iris_data.csv")
print (mydata)

#pairploting
mysb.pairplot(mydata,hue ='Species')
myplot.show()

#statistics

x = mydata.iloc[:,0:3]
print(x)

y = mydata.Species
print("\n",y)

mymodel = LogisticRegression(C = 1e08)

mymodel = mymodel.fit(x,y)
print(mymodel)

pred = mymodel.predict(x)
print(pred)

print("\n",mymodel.coef_)

print("\n",mymodel.intercept_)

rsq = mymodel.score(x,y)
print("\n",rsq)

print("\n",round(rsq*100,2))

mse = mean_squared_error(y,pred)
print("\n",mse)

print("\n",mymodel.intercept_)

preprob = mymodel.predict_proba(x)
print(preprob)

predclass = mymodel.predict(x)
print(predclass)

mytable = mypd.crosstab(y,predclass)
print(mytable)

predprob = mypd.DataFrame(preprob, columns = ["Predicted 0","Predicted 1","Predicted 2"])
print(predprob)

predclass = mypd.DataFrame(predclass,columns = ["Predicate Class"])
myresult = mydata.join(predclass)
print(myresult)

myresult = myresult.join(predprob)
print(round(myresult.head(15),4))