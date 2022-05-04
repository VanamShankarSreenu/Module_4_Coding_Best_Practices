from gettext import install
import os,tarfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import argparse
import csv
from os.path import dirname as up

class ingest_data:
    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        self.HOUSING_PATH = os.path.join("datasets", "housing")
        self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        self.strat_train_set =  None
        self.strat_test_set = None
        self.housing =  None


    def fetch_housing_data(self):
        os.makedirs(self.HOUSING_PATH, exist_ok=True)
        tgz_path = os.path.join(self.HOUSING_PATH, "housing.tgz")
        urllib.request.urlretrieve(self.HOUSING_URL, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.HOUSING_PATH)
        housing_tgz.close()


    def load_housing_data(self,housing_path):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)
    

    def load_data(self):
        self.fetch_housing_data()
        housing = self.load_housing_data(self.HOUSING_PATH)
        self.housing=housing
        return housing
    
    def stratifiedsplit(self):
        housing  = self.load_data()
        #stratified split
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        #remove INCOME CAT column from test and train
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set


    def DataTransformation(self,housing):
        #pipeline fill null values,encode catagorical data
        housing_labels = housing["median_house_value"].copy()
        housing = housing.drop("median_house_value", axis=1)
        housing_num = housing.drop("ocean_proximity",axis=1)
        housing_cat = housing[["ocean_proximity"]]
        num_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy="median")),
            ('std_scaler',StandardScaler())
        ])
        num_attribs = list(housing_num.columns)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
            ('num',num_pipeline,num_attribs),
            ("cat",OneHotEncoder(),cat_attribs)
        ])
        housing_prepared = full_pipeline.fit_transform(housing)
        housing_prepared = np.c_[housing_prepared,housing_labels]
        cat_encoder = full_pipeline.named_transformers_["cat"]
        cat_1hot_attrib = list(cat_encoder.categories_[0])
        att = num_attribs+cat_1hot_attrib+['median_house_value']
        df = pd.DataFrame(data=housing_prepared,columns=att)
        print(df)
        return df


    def transform_train_test(self):
        self.df_train = self.DataTransformation(self.strat_train_set)
        self.df_test =  self.DataTransformation(self.strat_test_set)
        #get path of folder from command line argumentsSS
        parser = argparse.ArgumentParser()
        parser.add_argument('dirname')
        try:
            args = parser.parse_args()
            dir = str(args.dirname)
            os.makedirs(dir, exist_ok=True)
            self.df_train.to_csv(dir+'/train.csv')
            self.df_test.to_csv(dir+'/test.csv')
        except:
            dir = up(up(os.path.realpath(os.getcwdb())))
            dir = dir.decode('utf-8')
            dir = str(dir)+'\data\processed'
            self.df_train.to_csv(dir+'\\train.csv')
            self.df_test.to_csv(dir+'\\test.csv')
            
    def split_X_Y(self,data):
        Y = data[["median_house_value"]]
        X = data.drop("median_house_value",axis=1)
        return X.values,Y


