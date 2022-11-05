# 3-1
## 任务要求
使用回归模型对含有十一个特征的数据集进行训练，并利用测试集进行算法检验。并使用RMSE作为评价指标。
## 实验源码
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

#location of dataset
train_datapath = "Homework-3-1\Homework-3-1-train_data.csv"
train_data=pd.read_csv(train_datapath)
train_data=np.array(train_data.values.tolist())
print(train_data)
train_n=train_data[:,0]
train_x=train_data[:,1:11]
train_y=train_data[:,12]

test_datapath = "Homework-3-1\Homework-3-1-test_data.csv"
test_data=pd.read_csv(test_datapath)
test_data=np.array(test_data.values.tolist())
test_n=test_data[:,0]
test_x=test_data[:,1:11]
test_y=test_data[:,12]
print(test_y)

def regression(model,str):
    #拟合预测
    model.fit(train_x,train_y)
    result=model.predict(test_x)
    #画图
    plt.figure()
    plt.scatter(test_n,test_y,s=5,label="true value")
    plt.scatter(test_n,result,s=5,label="regression value",color='r')
    plt.legend()
    plt.show()
    #计算根均方根RMSE
    test_rmse=mean_squared_error(result,test_y)**0.5
    print(str+"Prediction RMSE:{:.4f}".format(test_rmse))

## 比对多种模型
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

#main function
if __name__ == "__main__":
    #回归模型
    model=LinearRegression()
    regression(model,"LinearRegression")
    model=KNeighborsRegressor()
    regression(model,"KNeighborsRegressor")
    model=SVR()
    regression(model,"SVR")
    model=Lasso()
    regression(model,"Lasso")
    model=Ridge()
    regression(model,"Ridge")
    model=MLPRegressor()
    regression(model,"MLPRegressor")
    model=DecisionTreeRegressor()
    regression(model,"DecisionTreeRegressor")
    model=ExtraTreeRegressor()
    regression(model,"ExtraTreeRegressor")
    model=RandomForestRegressor()
    regression(model,"RandomForestRegressor")
    model=AdaBoostRegressor()
    regression(model,"AdaBoostRegressor")
    model=GradientBoostingRegressor()
    regression(model,"GradientBoostingRegressor")
    model=BaggingRegressor()
    regression(model,"BaggingRegressor")

```
## 实验结果
通过比对12种回归模型的预测结果，RandomForestRegressor随机森林回归模型的预测效果最好，RMSE最小，为0.5758
预测结果如下所示
![output](https://pyrojewelpicgo-1308141986.cos.ap-beijing.myqcloud.com/202211052106479.png)

# 3-2
## 任务要求
任务二为二分类模型，因此选择采用SVM模型进行训练，使用准确率作为评价指标。
## 实验源码
```python
#SVM二分类向量机
from sklearn import svm
def SVM():
    #获得数据
    train_data=pd.read_csv(r"Homework-3-2\Homework-3-2-train_data.csv")
    train_data=np.array(train_data.values.tolist())
    train_n=train_data[:,0]
    train_x=train_data[:,1:5]
    train_y=train_data[:,6]
    
    test_data=pd.read_csv(r'Homework-3-2\Homework-3-2-test_data.csv')
    test_data=np.array(test_data.values.tolist())
    test_n=test_data[:,0]
    test_x=test_data[:,1:5]
    test_y=test_data[:,6]
    print(test_y)
    
    #拟合预测
    model=svm.SVC(kernel='linear',C=1)
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
    #计算ROC曲线
    createROC(test_y,result)
if __name__ == "__main__":
    SVM()
```

## 实验结果

![output3-2_1](https://pyrojewelpicgo-1308141986.cos.ap-beijing.myqcloud.com/202211052109746.png)

![output3-2-2](https://pyrojewelpicgo-1308141986.cos.ap-beijing.myqcloud.com/202211052109755.png)

在测试集上获得了84%的准确率
