import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata = pd.read_csv("big.csv")
ge_le = LabelEncoder()
bo_le = LabelEncoder()
mydata["gender_encoded"] = ge_le.fit_transform(mydata["Gender"])
mydata["body_type_encoded"] = bo_le.fit_transform(mydata["Body Type"])
x = mydata[["Age", "gender_encoded", "body_type_encoded", "Height"]]
y = mydata["Weight"]
model = lm.LinearRegression()
model.fit(x, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
getAge = int(input("Enter age in years: "))
getGender = input("Enter gender: ")
getBodyType = input("Enter body type: ")
getHeight = int(input("Enter height in cm: "))
weight = model.predict([[getAge, ge_le.transform([getGender])[0], bo_le.transform([getBodyType])[0], getHeight]])
print("Your predicted weight:", weight)