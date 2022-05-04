import argparse
import logging
import os
from os.path import dirname as up
import pickle
from statistics import mode
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import numpy as np


class parse_log:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_folder", type = str,help="provide model folder")
        parser.add_argument("--dataset_folder", type = str,help="provide the dataset folder")
        parser.add_argument("--output_folder", type = str,help="provide the output folder")
        parser.add_argument("--log_level", type = str,help="specifiy level of debug")
        parser.add_argument("--log_path", type = str,help="specify the path where to save log file")
        parser.add_argument("--no_console_log", type = str,help="specify to log on console or not")
        args = parser.parse_args()
        self.args = args
        #getting logger object
        logger = logging.getLogger(__name__)
        #specifing the format we want to log
        formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
        #default setting to Debug level
        level = logging.DEBUG
        logger.setLevel(level)
        if args.log_level:
            if args.log_level == 'DEBUG':
                level=logging.DEBUG
            
            elif args.log_level == 'INFO':
                level=logging.INFO
                
            elif args.log_level == 'ERROR':
                level=logging.ERROR
            
            elif args.log_level == 'WARNING':
                level=logging.WARNING
            else:
                level=logging.CRITICAL
            logger.setLevel(level)

        if args.log_path:
            file_handler = logging.FileHandler(args.log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if not args.no_console_log:
            stream_handler = logging.StreamHandler()
            #set the console format
            stream_handler.setFormatter(formatter)
            #add console handler to logger
            logger.addHandler(stream_handler)
        logger.debug('INSIDE LOG')
        self.logger = logger
    

class scores:
    def __init__(self):
        obj = parse_log()
        self.args = obj.args
        self.logger = obj.logger
    
    def results(self):
        dir = up(os.path.realpath(os.getcwdb()))
        dir = dir.decode('utf-8')
        if self.args.dataset_folder:
            df = pd.read_csv(self.args.dataset_folder)
            Y = df[["median_house_value"]]
            X = df.drop("median_house_value",axis=1)
        else:
            df = pd.read_csv(str(dir) +'\\data\\processed\\test.csv')
            Y = df[["median_house_value"]]
            X = df.drop("median_house_value",axis=1)

        if self.args.model_folder:
            for filename in os.scandir(self.args.model_folder):
                with open(filename , 'rb') as f:
                    model = pickle.load(f)
                    pred = model.predict(X)
                    mse_ = mse(pred,Y)
                    rmse = np.sqrt(mse_)
                    r2 = r2_score(pred,Y)
                    self.logger.debug("rmse %d",rmse)
                    self.logger.debug("r2 %f",r2)
        else:
            for filename in os.scandir(str(dir)+'\data\models'):
                with open(filename , 'rb') as f:
                    model = pickle.load(f)
                    pred = model.predict(X)
                    mse_ = mse(pred,Y)
                    rmse = np.sqrt(mse_)
                    r2 = r2_score(pred,Y)
                    self.logger.debug("rmse %d",rmse)
                    self.logger.debug("r2 %f",r2)

obj = scores()
obj.results()
