# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:29:51 2018

@author: Evangelista
"""

from alpha_vantage.timeseries import TimeSeries
import time
import json
import pandas as pd

def download_json_data(companies, mode):    
    ts = TimeSeries(key='8T5R3MKIOEJGAL8F', output_format='json', retries = 10)    
    data, meta_data = {}, {}
    for company in companies:
        if mode == 'daily':
            data[company], meta_data[company] = ts.get_daily_adjusted(symbol=company, outputsize='full')
        elif mode == 'monthly':
            data[company], meta_data[company] = ts.get_monthly_adjusted(symbol=company)
        elif mode == 'intraday':
            data[company], meta_data[company] = ts.get_intraday(symbol=company, interval ='1min', outputsize='full')
        elif mode == 'batch':
            data[company], meta_data[company] = ts.get_batch_stock_quotes(companies)
            return data
        else:
            print("Only supported modes are 'daily','monthly', 'intraday', 'batch'")
            return data
        print("Finished: " + company)
        time.sleep(12)
    return data

def save_to_csv(data):
    for company in data: 
        expl = json.dumps(data[company])
        expl = pd.read_json(expl).T
        expl.to_csv("data/" + company + ".csv", index_label='date')
        del expl
        
'''Example usage    '''   
if __name__ == "__main__":
    '''downloading data'''
    companies = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'XOM', 'HD','IBM', 'INTC', 'JNJ',
                 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'WBA', 'DIS']
    '''deleted companies : DWDP 313, GS 4923, V 2691'''
    '''columns are open, high, low, close, adjusted close, volume, divident amount, split coefficient'''
    companies=['MMM']
    data = download_json_data(companies,'daily')
    #save_to_csv(data)