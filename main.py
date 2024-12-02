import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#declaring variables
age = 25
name = "Alice"
height = 5.4

#if, elif, and else statements
if (age >= 18):
    print("You are an adult")
else:
    print("You are a minor")

#Lists
numbers = [1,2,3,4,5,6,7,8,9,10]
print(numbers[1])

#Dictionaries
student = {"name": "John", "age": 20, "grade": "A"}
print(student["age"])

#loops
for i in range(5):
    print(i)

#functions
def greet(name):
    return f"Hello, {name}!"

print(greet(student["name"]))

#working with libraries
data = np.array([1,2,3,4,5])
print(data.mean())

#working with files
#file = open("exampleFile.txt","x")
with open("exampleFile.txt","w") as file:
    file.write("Hello, World!")

#working with datasets

dataset = pd.DataFrame({
    "Year":[2000,2001,2002],
    "Population": [2.3,2.5,2.7]
})

print(dataset)

#using the plot function along with visualization
x = [1,2,3,4,5]
y = [2,4,6,8,10]
plt.plot(x,y)
plt.show()

#using the math library
print(math.sqrt(16))