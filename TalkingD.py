import pickle
import sys
import numpy
import matplotlib
import pandas
import scipy
import seaborn
import sklearn
import datetime
print('Python:{}'.format(sys.version))
print('Numpy.{}'.format(numpy.__version__))
print('Pandas.{}'.format(pandas.__version__))
print('Matplotlib.{}'.format(matplotlib.__version__))
print('Scipy.{}'.format(scipy.__version__))
print('Seaborn.{}'.format(seaborn.__version__))
print('Sklearn.{}'.format(sklearn.__version__))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('train_sample.csv')
data.head()



Fraud = data[data['is_attributed'] == 1]
Valid = data[data['is_attributed'] == 0]
outlier_fraction = len(Fraud) / len(Valid)
print(outlier_fraction)
print('Fraud cases:{}'.format(len(Fraud)))
print('valid cases:{}'.format(len(Valid)))

percentage = (data.is_attributed.values == 0).mean()
plot = sns.barplot(['not downloaded','downloaded'],[percentage*100,(1-percentage)*100])
# display the proponataly of the data through bar graph
plot.set(ylabel ='Proportion')
for i in range(2):
    a = plot.patches[i]
    height = a.get_height()
    value = abs(percentage-i)
    plot.text(a.get_x()+a.get_width()/2., height+0.5, round(value*100, 2), ha="center")
plt.show()

# displays the unique values for features
unique_values = []

for x in data.columns:
    unique_values.append(len(data[x].unique()))
    unique_values.pop()

plot = sns.barplot(['ip', 'app', 'device', 'os', 'channel', 'clic_time', 'attri_time'], unique_values)
plot.set(ylabel='Unique Values')

for i in range(len(unique_values)):
    a = plot.patches[i]
    height = a.get_height()
    value = abs(percentage - i)
    plot.text(a.get_x()+a.get_width()/2., height+1, unique_values[i], ha="center")
plt.show()

# downloaded
ip_1= data.ip[data.is_attributed == 1]
set_of_ip_1= set(ip_1.unique())
ip_0= data.ip[data.is_attributed == 0]
set_of_ip_0= set(ip_0.unique())

ip_download = set_of_ip_1 - set_of_ip_0

ip_fraudulent =set_of_ip_0 -set_of_ip_1
print('the total no of ip through which the app was always downloaded: ', len(ip_download))
print('the total no of ip through which the app was never downloaded: ',len(ip_fraudulent))



print('the percentage of ips through which the app was always downloaded is: ',round ((len(ip_download)/277396)*100,2),"%")
print('the percentage of ips through which the app was always downloaded is: ',round ((len(ip_download)/277396)*100,2),"%")






#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax= .8, square= True)
plt.show()

#get all the coloumns from the dataframe
columns = data.columns.tolist()

#filter the coloumns to remove the data we do not need
columns = [c for c in columns if c not in ["is_attributed"]]

#store the variable we'll be predicting on
target = "is_attributed"

X = data[columns]
Y = data[target]

#print the shapes of x and y
print(X.shape)
print(Y.shape)

#applying the algorithms to the project
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#define random state
state = 1


#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)
}
model_file = "model.pkl"
#fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

#reshape the prediction values to 0 for valid and 1 for fraudulent
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        n_errors = (y_pred != Y).sum()

#run classification metrics
        print('{} : {}'.format(clf_name, n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y, y_pred))
        #print(np.array(y_pred))


#using pickle
pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))





