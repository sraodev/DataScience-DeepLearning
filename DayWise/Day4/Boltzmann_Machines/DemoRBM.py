# -*- coding: utf-8 -*-
"""
Predict the binary outcome yes or no 
for a movie to recommend it to a user
"""
# importing the libraries

import numpy as np   # numpy  for working with arrays
import pandas as pd  # for importing dataset & creating training & testing set
import torch 
#import torch.nn as nn   # Module of torch to implement neural networks
#import torch.nn.parallel   # Module for parallel computation  
#import torch.optim as optim  # Module for Optimizer
#import torch.utils.data
#from torch.autograd import Variable  # Module for Stochastic Gradient descent



# importing dataset
# All movies dataset first

# movie-id will be used for our recommender system 
# we wont use movies dataframe for training , its just to keep a track of movies in the dataset

# MovieID::Title::Genres
movies = pd.read_csv("ml-1m/movies.dat", sep='::',header=None,
                     engine="python",encoding="latin-1" )

# UserID::Gender::Age::Occupation::Zip-code
users = pd.read_csv("ml-1m/users.dat", sep='::',header=None,
                     engine="python",encoding="latin-1" )

# UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv("ml-1m/ratings.dat", sep='::',header=None,
                     engine="python",encoding="latin-1" )

"""
This data set consists of:
	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
	* Each user has rated at least 20 movies. 
        * Simple demographic info for the users (age, gender, occupation, zip)
        
u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
u1.test       are 80%/20% splits of the u data into training and test data.        

"""        
# preparing training set , 80%
# UserID::MovieID::Rating::Timestamp
training_set = pd.read_csv("ml-100k/u1.base",delimiter="\t")

mtraining_set = pd.read_csv("ml-100k/u1.base",delimiter="\t")
mtest_set = pd.read_csv("ml-100k/u1.test",delimiter="\t")

# converting Dataframe training set to a numpy array of integers
training_set = np.array(training_set,dtype="int64")

# preparing training set , 20%
# UserID::MovieID::Rating::Timestamp

test_set = pd.read_csv("ml-100k/u1.test",delimiter="\t")

# converting Dataframe testing set to a numpy array of integers
test_set = np.array(test_set,dtype="int64")



# capture the number of users and number of movies 
nb_users = 943
nb_movies = 1682

# we need to convert the data in a way that RBM expect the data
# RBM is a type of neural network which takes inputs 
# So we are creating an array out of training and testing set
# where the rows are Users and Columns are movies
# observations in lines and features in columns
# 943 rows and 1682 columns
# Funcion to convert training and testing to nested list
# where each child list corresponds to single user along with the ratings for the movie they saw
# for movies that they didnt see a zero will be added
# number of elements in the list for each user will be 1682
# nested list as we will create a torch tensor


def convert(data):
    new_data=[]
                         
    for id_user in range(1,nb_users+1):
                    
        id_movies = data[data[:,0]==id_user, 1] # watched movie-id for that user in id_user
        id_ratings = data[data[:,0]==id_user,2] # ratings for the watched movies for that user in id_user
        ratings = np.zeros(nb_movies)  # matrix, ratings for all 1682 movies watched as well as not watched ,filled with zero
        # overlap ratings for viewed moves and not viewed keep it zero
        #id_movies-1 ensures that we use index 0 also
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# now the lines are users and columns are movies
training_set=convert(training_set)    
test_set=convert(test_set)    


# pytorch tensors is a multi-dimensional arrays with same datatype
# pytorch tensors are better than numpy arrays and tensorflow tensors
# converting the above training_set & test_set to pytorch tensor

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# This RBM implementation is to predict whether the user liked the movie or not
# hence its going to be binary prediction
# we will convert all trainings into 0(not liked) & 1 (liked) and 0 ratings to -1 to indicate that the user didnt rate this movie as he didnt see it
# RBM will predict a binary output 0(not liked) & 1 (liked)
# RBM will take input vector and inside it will predicted the ratings for the movies that were not originally rated by the user
# since these predicted ratings are computed originally from the existing ratings from the input vector
# then the predicted ratings in the output must have the same format as the existing ratings in the input
# otherwise things would be inconsistant for the RBM

training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set >=3 ] = 1

test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set >=3 ] = 1






# =============================================================================
# Constructing the architecture of RBM
# Weight initialization for both hidden and visible nodes
# using randn() function
# self.W is a tensor matrix of rows as Weights of hidden node and columns as Weights of visible nodes           
# self.a is bias for hidden nodes probabilities of the hidden nodes given the visible nodes. p(h|v)
# self.b is bias for visible nodes probabilities of the visible nodes given the hidden nodes. p(v|h)
# for both a & b bias its must be a 2d because we are using functions that accept 2d tensors
# so while creating the bias tensor the shape is (1,nv) or (1,nh) , where 1 corresponds to batch

# sample_h
# wx = weight * visibleNeuron
# activation = wx + biasHidden
# p_h_given_v = sigmoid(activation)
# bernoulli_distribution(p_h_given_v)    

# sample_v
# wy = weight * hiddenNeuron
# activation = wy + biasVisible
# p_v_given_h = sigmoid(activation)
# bernoulli_distribution(p_v_given_h)  

# train function (Contrastive Divergence)
#The update of the weight matrix happens during the Contrastive Divergence step.
#Vectors v_0 and v_k are used to calculate the activation probabilities for hidden values h_0 and h_k

# new Weights = oldWeights + (v0*p(h0|v0) - vk*p(hk|vk))
# (v0*p(h0|v0) - vk*p(hk|vk)) is also know as DeltaWeight
# new Weights = oldWeights + DeltaWeight

# =============================================================================

class RBM():
    def __init__(self,nv,nh):
        
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)  # hidden node bias
        self.b=torch.randn(1,nv) # visible node bias
    
    def sample_h(self,x):
        wx = torch.mm(x, self.W.t())  # as we are calculating p(h|v) transpose needed
        activation = wx+self.a.expand_as(wx) 
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy = torch.mm(y,self.W)  # as we are calculating p(v|h) transpose not needed
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)    
   
    def train(self,v0,vk,ph0,phk):
        
        self.W = self.W + (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk) ).t()
        self.b = self.b + torch.sum((v0-vk),0) # visible node bias update
        self.a = self.a + torch.sum((ph0-phk),0) # hidden node bias update
      
# =============================================================================
# Calling the RBM class
# =============================================================================

nv=nb_movies
nh=100
batch_size=100

rbm=RBM(nv,nh)

# =============================================================================
# Training the Model
# train_loss for measuring the loss i.e. actual-predicted
# s is a counter which we will use to normalize the loss

# =============================================================================
            
nb_epoch = 10
  
        
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.                
    for id_user in range(0, nb_users - batch_size, batch_size):
                          
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        
        phk,_ = rbm.sample_h(vk)
        
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    





# Testing the RBM
test_loss = 0
s = 0.
total=0
size=0.

    
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    
    if len(vt[vt>=0]) > 0:
        
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
        total +=torch.sum(vt[vt>=0]==v[vt>=0],0) # number of correct predictions
        size += len(vt[vt>=0])  # total predictions

       
print('test loss: '+str(test_loss/s))
total = float(total)
pct = total/size
print("Percentage ",pct)        
        
# =============================================================================
# Single user prediction
# =============================================================================
for id_user in range(1):
    vOrig = training_set[id_user:id_user+1]
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    
    if len(vt[vt>=0]) > 0:
        
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
            

print(vOrig[0][9])   # 9th index ~ movie_id 10, user didnt see it
print(v[0][9])       # prediction by RBM
print(vt[0][9])      # actual rating of movie_id 10

print(vOrig[0][0])   # 0th index ~ movie_id 1, user didnt see it
print(v[0][0])       # prediction by RBM
print(vt[0][0])      # actual rating of movie_id 1 is not available
                        

print(vOrig[0][73])   # 73rd index ~ movie_id 74, user didnt see it
print(v[0][73])       # prediction by RBM
print(vt[0][73])      # actual rating of movie_id 74









# test set is same user as training set , in the same order, but with different subsets of the movies that user had rated.

"""
train_set  X_test
movies  m1 m2 m3 m4 m5 m6 m7 m8 m9 m10
ratings 1  2  1  3     1     5      3

test_set  y_test
movies  m1 m2 m3 m4 m5 m6 m7 m8 m9 m10
ratings             2     3     5

"""




























