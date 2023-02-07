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

mydata = mypd.read_csv("/home/nasc/Documents/G/ML/doc/Wine.csv")
print (mydata)


x = mydata.iloc[:,9:11]
print(x)

y = mydata.iloc[:,12:13]
print(y)

"""
#pairploting
mysb.pairplot(mydata,hue ='quality')
myplot.show()
"""

mymodel = LogisticRegression()
mymodel = mymodel.fit(x,y)
print(mymodel)
