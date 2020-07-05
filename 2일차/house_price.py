import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.mlab as mlab
from scipy import stats
from scipy.stats import norm, skew

import os
if not os.path.exists('./output'):
  os.mkdir('./output')

train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

train_ID = train_df['Id']
test_ID = test_df['Id']

train_df.drop('Id',axis=1,inplace=True)
test_df.drop('Id',axis=1,inplace=True)

#Outlier란 잘못 평가된 값으로 Univariate Multivariate
#Univariate란 변수 분포 하나에서 나타나는 Outlier이고
#Multivariate란 여러 변수 분포에서 나타나는 Outlier다

fig,ax = plt.subplots()
ax.scatter(x=train_df["GrLivArea"],y=train_df['SalePrice'])
plt.ylabel('SalePrice',fontsize=13)
plt.xlabel('GrLivArea',fontsize=13)
plt.show()

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000)&(train_df['SalePrice']<300000)].index)

fig,ax = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.ylabel('SalePrice',fontsize=13)
plt.xlabel('GrLivArea',fontsize=13)
plt.show()

sns.distplot(train_df['SalePrice'],fit=norm)
(mu,sigma) = norm.fit(train_df['SalePrice'])
print('\n mu ={:.2f} and sigma = {:.2f}\n'.format(mu,sigma))
#mu는 평균을 말하고 sigma는 표준 편차를 말한다

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.distplot(train_df['SalePrice'],fit=norm)
(mu,sigma) = norm.fit(train_df['SalePrice'])
print('\nmu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train_df['SalePrice'],plot=plt)
plt.show()

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df,test_df)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace= True)

combine_na = (all_data.isnull().sum() / len(all_data)) * 100
combine_na = combine_na.drop(combine_na[combine_na==0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' : combine_na})

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x : x.fillna(x.median()))

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType', 'MSSubClass', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1', 
            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

all_data.drop(['Utilities'], axis=1,inplace=True)
all_data["Functional"] = all_data["Functional"].fillna("Typ")

combine_na = (all_data.isnull().sum() / len(all_data)) *100
combine_na = combine_na.drop(combine_na[combine_na==0].index,axis=0).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' : combine_na})
#print(missing_data.head())

corrmat = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.9,square=True)
plt.show()

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
  lbl = LabelEncoder()
  lbl.fit(list(all_data[c].values))
  all_data[c] = lbl.transform(list(all_data[c].values))
#print('Shape all_data : {}'.format(all_data.shape))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness = skewness[abs(skewness)>0.75]
#print("There are{} skewed numerical features to BoxCox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_feature = skewness.index
lam = 0.15
for feat in skewed_feature:
  all_data[feat] = boxcox1p(all_data[feat],lam)

all_data = pd.get_dummies(all_data, drop_first=True)

train = all_data[:ntrain].values
test = all_data[ntrain:].values

X_train,X_val,y_train,y_val = train_test_split(train,y_train,test_size=0.1)
print(X_train.shape)
print(X_val.shape)
print(y_val.shape)
print(y_train.shape)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch.manual_seed(1234)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 216)
        self.fc2 = nn.Linear(216, 512)
        self.fc3 = nn.Linear(512, 216)
        self.fc4 = nn.Linear(216, 72)
        self.fc5 = nn.Linear(72, 18)
        self.fc6 = nn.Linear(18, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        return x

model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = 100
n_epochs = 1000
batch_no = len(X_train) // batch_size

val_loss_min = np.inf
for epoch in range(n_epochs):
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.FloatTensor(y_train[start:end]).squeeze())

        optimizer.zero_grad()
        output = model(x_var)
        loss = torch.sqrt(criterion(torch.log(output.squeeze()), torch.log(y_var)))
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        X_val = Variable(torch.FloatTensor(X_val))
        y_val = Variable(torch.FloatTensor(y_val))
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val.squeeze())

        if val_loss < val_loss_min:
            print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(val_loss_min, val_loss))
            torch.save(model.state_dict(), "./output/model.pt")
            val_loss_min = val_loss

        print('')
        print("Epoch: {} \tValidation Loss: {}".format(epoch+1, val_loss))

print('Training Ended! ')

model.load_state_dict(torch.load("./output/model.pt"))
test = Variable(torch.FloatTensor(test))
result = np.expm1(model(test).data.squeeze().numpy())

submission = pd.DataFrame()
submission["Id"] = test_ID
submission["SalePrice"] = result
submission.to_csv('./output/submission.csv',index=False)












