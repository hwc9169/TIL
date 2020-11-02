import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch.manual_seed(0)


import os
if not os.path.exists('./output'):
    os.mkdir('./output')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.drop(['Ticket','Cabin'],axis=1,inplace=True)
test_df.drop(['Ticket','Cabin'],axis=1,inplace=True)
combine =[train_df,test_df]

for dataset in combine:
     dataset['Title']  = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
for dataset in combine:
     dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
     dataset['Title'] = dataset['Title'].map(title_mapping)
     dataset['Title'] = dataset['Title'].fillna(0)

train_df.drop(['Name','PassengerId'],axis=1,inplace=True)
test_df_id = test_df['PassengerId']
test_df.drop(['Name','PassengerId'],axis=1,inplace=True)

sex_mapping = {'female':1,'male':0}
for dataset in combine:
     dataset['Sex'] = dataset['Sex'].map(sex_mapping)

guess_ages = np.zeros((2,3))
for dataset in combine:
     for i in range(0,2):
          for j in range(0,3):
               guess_df = dataset.loc[(dataset['Sex']==i) & (dataset['Pclass']==j+1),'Age'].dropna()

               guess_ages[i,j] = int(guess_df.median())

     for i in range(0, 2):
          for j in range(0, 3):
               dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1),'Age']=guess_ages[i,j]
     dataset['Age']=dataset['Age'].astype(int)

train_df['AgeBand']=pd.cut(train_df['Age'],5)

for dataset in combine:
     dataset.loc[dataset.Age<=16,'Age'] = 0
     dataset.loc[(dataset.Age <=32) &(dataset.Age > 16), 'Age'] = 1
     dataset.loc[(dataset.Age <=48) &(dataset.Age > 32), 'Age'] = 2
     dataset.loc[(dataset.Age <= 64) & (dataset.Age > 48), 'Age'] = 3
     dataset.loc[(dataset.Age > 64), 'Age'] = 4


train_df.drop('AgeBand' ,axis=1,inplace=True)
for dataset in combine:
    dataset['FamilySize'] = dataset['Parch']+dataset['SibSp']+1

for dataset in combine:
     dataset['IsAlone']=0
     dataset.loc[dataset['FamilySize']==1,'IsAlone'] = 1

train_df.drop(['Parch','SibSp','FamilySize'],axis=1,inplace=True)
test_df.drop(['Parch','SibSp','FamilySize'],axis=1,inplace=True)

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
     dataset.loc[dataset.Embarked.isnull(),'Embarked'] = freq_port

embarked_mapping={'S':0,'C':1,'Q':2}
for dataset in combine:
    dataset['Embarked']= dataset['Embarked'].map(embarked_mapping).astype(int)

train_df['Fare'].fillna(train_df['Fare'].dropna().median(),inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'],4)


for dataset in combine:
     dataset.loc[dataset.Fare<=7.91, 'Fare'] = 0
     dataset.loc[(dataset.Fare > 7.91)&(dataset.Fare<= 14.454), 'Fare'] = 1
     dataset.loc[(dataset.Fare> 14.454)&(dataset.Fare <= 31.0), 'Fare'] = 2
     dataset.loc[(dataset.Fare>31.0), 'Fare'] = 3

train_df.drop(['FareBand'],axis=1,inplace=True)
X_train = train_df.iloc[:,1:].values
y_train = train_df.iloc[:, :1].values
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#X_train train의 input
#X_val validation의 input
#y_train train의 label
#y_val validation의 label


torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x - self.dropout(x)
        x = self.fc4(x)
        return x

model = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

batch_size = 64
n_epochs = 1000
batch_no = len(X_train) // batch_size

X_val = Variable(torch.FloatTensor(X_val)).cuda()
y_val = Variable(torch.LongTensor(y_val)).cuda()

val_loss_min = np.inf
for epoch in range(n_epochs):
    for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size

            x_var = Variable(torch.FloatTensor(X_train[start:end])).cuda()
            y_var = Variable(torch.LongTensor(y_train[start:end]).squeeze()).cuda()

            optimizer.zero_grad()
            output = model(x_var)
            loss = criterion(output, y_var)
            loss.backward()
            optimizer.step()

    val_output = model(X_val)
    val_loss = criterion(val_output, y_val.squeeze())

    if val_loss <= val_loss_min:
        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(val_loss_min, val_loss))
        torch.save(model.state_dict(), "./output/model.pt")
        val_loss_min = val_loss

    if epoch % 100 == 0:
        values, labels = torch.max(val_output, 1)
        # num_right = np.sum(labels.data.numpy() == y_val.squeeze().data.numpy())

        print('')
        print("Epoch: {} \tValidation Loss: {}".format(epoch+1, val_loss))#, num_right / len(y_val)))

print('Training Ended! ')

# test dataset에는 survived column이 없기 때문에 iloc[:,1:]를 할 필요가 없어. - 이희웅

X_test = test_df.values
X_test = Variable(torch.FloatTensor(X_test)).cuda()
with torch.no_grad():
    test_result = model(X_test)
values, labels = torch.max(test_result, 1)
survived = labels.squeeze().cpu().numpy()

submission = pd.DataFrame({'PassengerId' : test_df_id, 'Survived': survived})
submission.to_csv('./output/submission.csv',index=False)


