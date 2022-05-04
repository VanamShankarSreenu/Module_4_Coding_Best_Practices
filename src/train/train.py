from ingest_data import ingest_data
from matplotlib.pyplot import cla
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class buildmodel():
    def __init__(self):
        getdata = ingest_data.ingest_data()
        getdata.load_data()
        getdata.stratifiedsplit()
        getdata.transform_train_test()
        self.train = getdata.df_train
        self.test = getdata.df_test
        self.X_train,self.Y_train = getdata.split_X_Y(self.train)
        self.X_test,self.Y_test = getdata.split_X_Y(self.test)
    

class LinReg:
    def __init__(self):
        val = buildmodel()
        self.X_train,self.Y_train = val.X_train,val.Y_train
        self.X_test,self.Y_test = val.X_test,val.Y_test
        
    def train(self):
       lr = LinearRegression()
       lr.fit(self.X_train, self.Y_train)


class DesTree:
    def __init__(self):
        val = buildmodel()
        self.X_train,self.Y_train = val.X_train,val.Y_train
        self.X_test,self.Y_test = val.X_test,val.Y_test
        
    def train(self):
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.X_train, self.Y_train)
       
    def gridsearchcv(self):
        pass 


class RanFor:
    def __init__(self):
        val = buildmodel()
        self.X_train,self.Y_train = val.X_train,val.Y_train
        self.X_test,self.Y_test = val.X_test,val.Y_test
        
    def train(self):
        tree_reg = RandomForestRegressor()
        tree_reg.fit(self.X_train, self.Y_train)
       
    def gridsearchcv(self):
        pass 

