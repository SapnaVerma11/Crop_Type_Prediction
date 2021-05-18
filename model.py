# Importing the libraries
import pandas as pd
import pickle


#reading dataset
df = pd.read_csv('Crop_recommendation.csv')
df.shape

df.info()



#seperating dependent and independent variables
x = df.iloc[:,:-1].values
y = df.iloc[::,-1].values


# splitting dataset into traina and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 1)


# applying KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# Performing Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
k_range = list(range(1,31))
weight_options = ["uniform","distance"]
metric_options = ["euclidean","manhattan"]
param_grid = dict(n_neighbors = k_range,weights = weight_options,metric = metric_options)

grid = GridSearchCV(knn,param_grid,cv = 8,verbose = 1,n_jobs = -1)
grid.fit(x,y)

print(grid.best_params_)


classifier = KNeighborsClassifier(n_neighbors=5,metric="manhattan")
classifier.fit(x_train,y_train)
ypredict = classifier.predict(x_test)





from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,ypredict)
acc

pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[90,42,43,20.8,82.0,6.50,202.9]]))