import numpy as np 
from collections import Counter

from matplotlib import pyplot as plt

class perceptron:
    def __init__(self,input,weight=None):
        if weight==None:
            self.weight=np.random.random((input))*2-1
            print(input)
            print("selfweight is:",self.weight)
        self.learning_Rate=0.1

    @staticmethod
    def unit_function(x):
        if x < 0:
            return 0
        return 1
    
    def __call__(self,data):
        weight_input=self.weight*data
        print("data is" ,data)
        weight_sum=weight_input.sum()
        print("weight sum is",weight_sum)
        return perceptron.unit_function(weight_sum)
    
     
    def adjust(self, 
               target_result, 
               calculated_result,
               in_data):
        error = target_result - calculated_result
       
     
        for i in range(len(in_data)):
            correction = error * in_data[i] * self.learning_Rate
            self.weight[i] += correction 
        print("error",error)
        print("target",target_result)
        print("cal",calculated_result)
def above_line(point, line_func):
    x, y = point
    if y > line_func(x):
        return 1
    else:
        return 0
    print("point",point)
    print("line",line_func)

points=np.random.randint(1,100,(100,2))
print("points",points)
p=perceptron(2)
def lin1(x):

    return  x + 4
    print(lin1)
for point in points:
    p.adjust(above_line(point, lin1), 
             p(point), 
             point)
evaluation = Counter()
for point in points:
    if p(point) == above_line(point, lin1):
        evaluation["correct"] += 1
    else:
        evaluation["wrong"] += 1
print(evaluation.most_common())


cls = [[], []]
for point in points:
    cls[above_line(point, lin1)].append(tuple(point))
colours = ("r", "b")
for i in range(2):
    X, Y = zip(*cls[i])
    plt.scatter(X, Y, c=colours[i])
    
X = np.arange(-3, 120)
    
m = -p.weight[0] / p.weight[1]
print(m)
plt.plot(X, m*X, label="ANN line")
plt.plot(X, lin1(X), label="line1")
plt.legend()
plt.show()




