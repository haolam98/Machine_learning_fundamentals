import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle

# save our best model to later use so we don't have to re-train over and over again
# esentially we want to save our model that have the high accuracy by using pickel
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#read data in
#sep ~ seperator. In cvs file, each data is seperated by ';'
data = pd.read_csv("student-mat.csv", sep=";")


#trim data dowm to only attributes we want: G1, G2, studytime, failure, absences
# -pick attribute with int value. If it's a string we need to convert it to int
# -(entire data has ~32 attributes, see details on UCI Data Set info)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


# print first 10 data
#print(data.head)

# set up label -> we want machine to determine/predict G3
predict = "G3"

# set up 2 arrays
# - 1 array will store our lable/lables
# - 1 array will store our attributes

# this returns a new dataframe that does NOT have G3 <- for later to train machine
x = np.array(data.drop([predict],1))

# return a new dataframe that only have G3 <- for later to compare with machine prediction
y= np.array(data[predict])

# taking our lables and attributes that we trying to predict, and split them into 4 different arrays
# - x_train is a portion of x ; y_train is a portion of y
# - x_test and y_test is used to test the accuracy of machine prediction
# it splits up 10% of our test data into test sample (x_test & y_test) to test the machine as it never seen that
# data before
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# best = 0
# time_train = 30 # trainning times. Could do more but time connsuming
#
# for _ in range(time_train):
#     x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
#
#     # create a training model
#     linear = linear_model.LinearRegression()
#
#     # find the best fit line of the training data
#     linear.fit(x_train,y_train)
#
#     # get the accuracy of the prediction. Check how well the algorithm is?
#     acc = linear.score(x_test, y_test)
#
#     #save the best model <- we only save BEST one (higher accuracy)
#     if acc > best:
#         best =acc
#         # -- save our model for later use
#         # create a pickle file for us in our directory that we can open and use that
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)


# read in our pickle file
pickle_in = open("studentmodel.pickle","rb")

# load pickle to our linear models
linear = pickle.load(pickle_in)

# get the accuracy of the prediction. Check how well the algorithm is?
acc = linear.score(x_test, y_test)

print ("The accuracy of prediction: ",acc) #output : 0.84 ~ 84% of accuracy
print ("Coefficient: ",linear.coef_)  # slopes of the linear in multi-dimension
print("Intercepts:",linear.intercept_) # b in y= am+ b

# get machine predict G3 on each student data on the test data (x_test,the portion we did not train)
predictions = linear.predict(x_test)

# print out prediction
for x in range(len(predictions)):
    print("predict result: ",predictions[x], "input dat: ",x_test[x], "actual result:",y_test[x])


# Plot <- see correlations we have between each attribute affect toward G3= final grade
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("G3=Final Grade")
pyplot.show()





