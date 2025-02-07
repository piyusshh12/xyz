import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("hpmarch24.csv")
print(data)

feature= data[["area"]]
target= data["price"]

model= LinearRegression()
model.fit(feature.values, target)

area= float(input("Enter area: "))
price= model.predict([[area]])
print(price)