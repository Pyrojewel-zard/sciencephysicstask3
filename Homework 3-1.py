import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

#获得数据
train_data=pd.read_csv(r"Homework-3-1-train_data.csv")
train_data=np.array(train_data.values.tolist())
train_n=train_data[:,0]
train_x=train_data[:,1:12]
train_y=train_data[:,12]

test_data=pd.read_csv(r'Homework-3-1-test_data.csv')
test_data=np.array(test_data.values.tolist())
test_n=test_data[:,0]
test_x=test_data[:,1:12]
test_y=test_data[:,12]
print(test_y)

#回归函数
def regression(model):
    #拟合预测
    model.fit(train_x,train_y)
    result=model.predict(test_x)
    #画图
    plt.figure()
    plt.scatter(test_n,test_y,s=5,label="true value")
    plt.scatter(test_n,result,s=5,label="regression value",color='r')
    plt.legend()
    plt.show()
    #计算根均方差RMSE
    test_rmse=mean_squared_error(result,test_y)**0.5
    print("Prediction RMSE:{:.4f}%".format(test_rmse))


####回归方法####
#随机森林回归
from sklearn import ensemble
model1 = ensemble.RandomForestRegressor(n_estimators=30)
regression(model1)
