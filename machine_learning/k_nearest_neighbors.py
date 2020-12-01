import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

# K nearest neighbors (KNN) works with classification datasets

# read in data
data = pd.read_csv("car.data")
#print(data.head())

# since our data is non-numerical - string
# => we need to transfer this String data into numerical data (int)
le = preprocessing.LabelEncoder()
# we turn our data to a list,
# then transform it into appropriate interger value
buying = le.fit_transform(list(data["buying"]))  #return a numpy array
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


#similar to Linear Regression, we...

# set up label -> we want machine to determine/predict 'class' of the car
predict = "class"

# set up 2 arrays
# this returns a new dataframe with ALL attribute EXCEPT 'class' <- for later to train machine
#zip will push all info of each car into tuple. We will have multiple tuples
x = list(zip(buying,maint,door,persons,lug_boot,safety))
# return a new dataframe that only have G3 <- for later to compare with machine prediction
y = list(cls)


# taking our lables and attributes that we trying to predict, and split them into 4 different arrays
# - x_train is a portion of x ; y_train is a portion of y
# - x_test and y_test is used to test the accuracy of machine prediction
# it splits up 10% of our test data into test sample (x_test & y_test) to test the machine as it never seen that
# data before
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#train model
k_neigbor = 5
model = KNeighborsClassifier(n_neighbors=k_neigbor)
model.fit(x_train,y_train)

# get accuracy
acc= model.score(x_test,y_test)
print("The accuracy of prediction: ",acc) #output : 0.84 ~ 84% of accuracy

predictions = model.predict(x_test)

# print out prediction
names = ["unacc","acc","good","vgood"] #classfier name

for x in range(len(predictions)):
    print("predicted result: ",predictions[x], "input dat: ",x_test[x], "actual result:",names[y_test[x]])
    # #get neighbor of each point
    # n = model.kneighbors([x_test[x]],9,True)
    # print("N", n)

