#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(tag,dates, eval_test, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=False):
    """
    update train log file
    Function modified from sample code provided for Hands on Lab.

    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','tag','start_date','end_date','eval_test','model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), tag, dates[0],dates[-1],eval_test,
                            MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=False):
    """
    update predict log file
    Function modified from sample code provided for Hands on Lab.
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','country','y_pred','y_proba','target_date',
              'model_version','runtime','']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(),country, y_pred, y_proba,target_date,
                            MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """
    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    ## train logger
    update_train_log('testing',(str(date.today()),str(date.today())),"{'rmse':0.5}","00:00:01",
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
    ## predict logger
    update_predict_log("USA","[0]", "[0.6,0.4]",str(date.today()),
                       "00:00:01", MODEL_VERSION, test=True)
    
        
