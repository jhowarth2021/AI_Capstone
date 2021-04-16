#!/usr/bin/env python
"""
model tests
"""

import sys, os
import numpy as np
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"cs-train")
        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test.joblib")))

    def test_02_load(self):
        """
        test the load functionality
        """              
        ## train the model
        model = model_load(os.path.join(os.path.dirname(os.path.dirname(__file__)),"cs-train"))
        
        self.assertIsNotNone(model)

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load(os.path.join(os.path.dirname(os.path.dirname(__file__)),"cs-train"))
    
        ## ensure that a list can be passed
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day)
        y_pred = result['y_pred']
        self.assertEquals(y_pred,y_pred)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
