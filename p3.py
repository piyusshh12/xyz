import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv("hpmarch24.csv")
print(data)

feature= data[["area"]]
target= data["price"]

print(feature)
print(target)

model= LinearRegression()
model.fit(feature.values, target)

plt.scatter(data["area"],data["price"],color="red")
plt.plot(feature,model.predict(feature.values),color="blue")
plt.xlabel("area")
plt.ylabel("price")
plt.show()