# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 12:18:54 2018

@author: Ammar
"""

import csv

import matplotlib.pyplot
import pylab
import numpy as np
import math
import copy

from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title


#reading the data from the file
with open('ex2data2.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

Features=[]

n=len(data[0])-1

#as many features as n-1
for i in range(0,n):
    Features.append([])


#populate the list using data
for i in range(0,len(data)):
    for j in range (0,n):
        Features[j].append(float(data[i][j]))
        
#Making and initialzing Theetas
Theetas=[]
for i in range (0,28):
    Theetas.append(float(0))
    
#separating the 0's and 1's
admittedBoolean=[]
for i in range(0,len(data)):
  admittedBoolean.append(float(data[i][n]))
  

#plotting the scatter plot
for i in range(0,len(data)):
    if admittedBoolean[i]==0:
        matplotlib.pyplot.scatter(Features[0][i],Features[1][i],c='red',marker='o')
    elif admittedBoolean[i]==1:
        matplotlib.pyplot.scatter(Features[0][i],Features[1][i],c='green',marker='x')

matplotlib.pyplot.xlabel("Microchip Test 1")
matplotlib.pyplot.ylabel("Microchip Test 2")


#normalizing the features

means=[]
stds=[]

for i in range(0,n):
    mean=np.mean(Features[i])
    std=np.std(Features[i])
    means.append(float(mean))
    stds.append(float(std))
    for j in range(len(Features[i])):
        Features[i][j]=(Features[i][j]-mean)/std  
        

#making the mapped feature
mappedFeature=[]
for i in range(0,27):
    mappedFeature.append([])
    
for i in range(0,len(data)):
    x1=Features[0][i]
    x2=Features[1][i]
    mappedFeature[0].append(float(x1))
    mappedFeature[1].append(float(x2))
    mappedFeature[2].append(float(x1*x1))
    mappedFeature[3].append(float(x1*x2))
    mappedFeature[4].append(float(x2*x2))
    mappedFeature[5].append(float(x1*x1*x1))
    mappedFeature[6].append(float(x1*x1*x2))
    mappedFeature[7].append(float(x1*x2*x2))
    mappedFeature[8].append(float(x2*x2*x2))
    mappedFeature[9].append(float(x1*x1*x1*x1))
    mappedFeature[10].append(float(x1*x1*x1*x2))
    mappedFeature[11].append(float(x1*x1*x2*x2))
    mappedFeature[12].append(float(x1*x2*x2*x2))
    mappedFeature[13].append(float(x2*x2*x2*x2))
    mappedFeature[14].append(float(x1*x1*x1*x1*x1))
    mappedFeature[15].append(float(x1*x1*x1*x1*x2))
    mappedFeature[16].append(float(x1*x1*x1*x2*x2))
    mappedFeature[17].append(float(x1*x1*x2*x2*x2))
    mappedFeature[18].append(float(x1*x2*x2*x2*x2))
    mappedFeature[19].append(float(x2*x2*x2*x2*x2))
    mappedFeature[20].append(float(x1*x1*x1*x1*x1*x1))
    mappedFeature[21].append(float(x1*x1*x1*x1*x1*x2))
    mappedFeature[22].append(float(x1*x1*x1*x1*x2*x2))
    mappedFeature[23].append(float(x1*x1*x1*x2*x2*x2))
    mappedFeature[24].append(float(x1*x1*x2*x2*x2*x2))
    mappedFeature[25].append(float(x1*x2*x2*x2*x2*x2))
    mappedFeature[26].append(float(x2*x2*x2*x2*x2*x2))
    
      
    
            
      
#function to find GofX
def findGOfX(TheetaTX):
  
    denom=1+math.exp(-(TheetaTX))
    
    return (1/denom)
    

#function to find hOfX for prediction purposes    
def find_h_of_x_for_Evaluation(testTheetas=[],testFeatures=[]):
    
    matTheetas=np.asmatrix(testTheetas)
    matFeatures=np.asmatrix(testFeatures)
    hofx=np.matmul(matTheetas,matFeatures.transpose())
    
    print(hofx)
   
    return hofx

#function to find hOfX for gradient descent purpose   
def find_h_of_x_at_Index(index,Theetas=[],Features=[]):
   
    x=[]

    x.append(float(1))

    for i in range(0,len(Features)):
        x.append(Features[i][index])
        
  
    matTheetas=np.asmatrix(Theetas)
    matFeatures=np.asmatrix(x)
    hofx=np.matmul(matTheetas,matFeatures.transpose())
   
    return hofx
    
    
   
#cost function
def costFunction(m,Theetas=[],Features=[],admittedBoolean=[]):
    cost=0
    cost2=0
    for i in range(0,m):
        firstTerm=(-1*admittedBoolean[i])*(math.log(findGOfX(find_h_of_x_at_Index(i,Theetas,Features)) ))
        secondTerm=(1-admittedBoolean[i])*(math.log(1-findGOfX(find_h_of_x_at_Index(i,Theetas,Features)) ))
        finalTerm=firstTerm-secondTerm
        cost=cost+finalTerm
    
    cost=(cost/m)
    
    for j in range(1,len(Theetas)):
        cost2=cost2+(Theetas[j]*Theetas[j])
        
    lambdaValue=1    
    cost2=(cost2*lambdaValue)/(2*m)
    
    return (cost+cost2)

#printing the initial cost with all theetas = 0
print("Initial cost with all theetas = 0")
print(costFunction(len(data),Theetas,mappedFeature,admittedBoolean))

#function to update theetas
def updateTheeta(k,m,Theetas=[],Features=[],admittedBoolean=[]):
   sum=0
   alpha=0.001
   for i in range(0,m):
       firstTerm=findGOfX(find_h_of_x_at_Index(i,Theetas,Features))
       secondTerm=admittedBoolean[i]
       
       x=[]

       x.append(float(1))

       for l in range(0,len(Features)):
           x.append(Features[l][i])
       
       
       finalTerm=(firstTerm-secondTerm)*x[k]
       sum=sum+finalTerm
       
   return ((alpha*sum)/m)


tempTheetas=copy.copy(Theetas)

#applying gradient descent and updating the theetas
for i in range(0,200):             
    for j in range(0,len(tempTheetas)):
        alpha=0.001
        lambdaValue=1
        tempTheetas[j]=(tempTheetas[j]*(1-((alpha*lambdaValue)/len(data))))-updateTheeta(j,len(data),Theetas,mappedFeature,admittedBoolean)
        
    Theetas=copy.copy(tempTheetas)
    tempTheetas=copy.copy(Theetas)
   
    
print("Cost after optimized theetas:")    
print(costFunction(len(data),Theetas,mappedFeature,admittedBoolean))
print("Optimized Theetas:")
print(Theetas)



   


#function that returns 6 degree array
def map_feature(x1, x2):
    
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = append(out, r, axis=1)

    return out

#Plotting the decision boundary line        
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (map_feature(np.array(u[i]), np.array(v[j])).dot(np.array(Theetas)))

z = z.T
contour(u, v, z)
show()
    


    

    
    
    
