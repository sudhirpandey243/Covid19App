import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
data = pd.read_csv('datasetREVAHackthon1230N.csv')
df1=data.dropna()
Y=df1.Risk
X=df1.drop('Risk',axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_test)

logreg.score(x_train, y_train), logreg.score(x_test, y_test)

import pickle
#logreg = LogisticRegression()
pickle.dump(logreg, open('model.pkl','wb'))
pickle.dump(logreg, open('model.pkl','wb'))
#with open('model_pickle','rb') as f:
    #model = pickle.load(f)
print(model.predict([[76,2,5,10,20,100,0,1]]))