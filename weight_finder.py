import pandas as pd
import sklearn.linear_model as lm
mydata = pd.read_csv("weight_predict_real.csv")
x = mydata[["height", "age", "bmi", "muscle_mass", "body_fat"]]
y = mydata["weight"]
model = lm.LinearRegression()
model.fit(x, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
getHeight = int(input("Enter height in cm: "))
getAge = int(input("Enter age in years: "))
getBmi = float(input("Enter BMI: "))
getMuscleMass = float(input("Enter muscle mass in kg: "))
getBodyFat = float(input("Enter body fat percentage: "))
print(model.predict([[getHeight, getAge, getBmi, getMuscleMass, getBodyFat]]))