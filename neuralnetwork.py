import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder




#loading csv to dataframe
df=pd.read_csv('breast-cancer.csv')
df.drop(columns=['id'], inplace=True)
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2)

#scaling the values
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

#encoding Y values
encode=LabelEncoder()
y_train=encode.fit_transform(y_train)
y_test=encode.transform(y_test)

#coverting to tensor
x_train_tensor=torch.from_numpy(x_train)
x_test_tensor=torch.from_numpy(x_test)
y_train_tensor=torch.from_numpy(y_train).float().unsqueeze(1)
y_test_tensor=torch.from_numpy(y_test).float().unsqueeze(1)



#creating the neuralnetwork
class MySimpleNN():
    def __init__(self,x):
        self.weights=torch.rand(x.shape[1],1, dtype=torch.float64, requires_grad=True)
        self.bias=torch.zeros((1), dtype=torch.float64, requires_grad=True)
    def forward_pass(self,x):
        z= torch.matmul(x,self.weights)+ self.bias
        y_Pred= torch.sigmoid(z)
        return y_Pred
    def loss(self,y,y_Pred):
        eps = 1e-8
        y_Pred = torch.clamp(y_Pred, eps, 1 - eps)

        ## -[y*log(p) + (1-y)*log(1-p)]
        bce_loss = - (y * torch.log(y_Pred + eps) + (1 - y) * torch.log(1 - y_Pred + eps))
        loss =bce_loss.mean()
        return loss



#defining learning rate
learning_rate=0.001
#epchs(no of iterations in a loop)
epochs= 50

#TRAINING PIPELINE

#defining model
model=MySimpleNN(x_train_tensor)

#defining loop
for epoch in range(epochs):
    #forward pass
    y_pred= model.forward_pass(x_train_tensor)
    #calculating loss
    loss=model.loss(y_train_tensor,y_pred)
    #backward pass (derivatives using autograd)
    loss.backward()
    #updating weights and biases
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad
    #resetting the grads
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #analyzing loss with epochs
   # print(f'epoch: {epoch+1}, loss: {loss.item()}')



#model evaluation
with torch.no_grad():
    y_predt = model.forward_pass(x_test_tensor)
    y_predt =(y_predt>0.5).float()
    accuracy= (y_predt==y_test_tensor).float().mean()
    print(f'model accuracy: {accuracy.item()}')



