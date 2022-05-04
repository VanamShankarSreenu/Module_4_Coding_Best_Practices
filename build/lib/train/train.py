from ingest_data import ingest_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


class buildmodel():
    def __init__(self):
        getdata = ingest_data.ingest_data()
        getdata.load_data()
        getdata.stratifiedsplit()
        getdata.transform_train_test()
        self.train = getdata.df_train
        self.test = getdata.df_test
        print(self.train.shape)
        self.X_train,self.Y_train = getdata.split_X_Y(self.train)
        print(self.X_train.shape)
        self.X_test,self.Y_test = getdata.split_X_Y(self.test)
    
class LinReg:
    def __init__(self):
        val = buildmodel()
        self.X_train,self.Y_train = val.X_train,val.Y_train
        self.X_test,self.Y_test = val.X_test,val.Y_test
        
    def train(self):
       lr = LinearRegression()
       print(self.X_train.shape)
       lr.fit(self.X_train, self.Y_train)
       pred = lr.predict(self.X_test)
       lin_mse = mse(self.Y_test,pred)
       print(lin_mse)
    
    def test(self):
        pass

    def gridsearchcv(self):
        pass

LinReg().train()

