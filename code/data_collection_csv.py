# Import the required modules 
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
import pandas as pd
import os
import webbrowser

# Read the csv into a pandas dataframe
df = pd.read_csv('code/data/data/constituents_csv.csv')

symbols = df['Symbol']

# ---------------------------- BE CAREFUL, WILL DOWNLOAD A BUNCH OF CSV FILES, BROWSER MAY CRASH ----------------------------------------
""" for symbol in symbols:
    webbrowser.open('https://query1.finance.yahoo.com/v7/finance/download/' + symbol + '?period1=1232841600&period2=1671926400&interval=1d&events=history&includeAdjustedClose=true')
 """
# ----------------------------------------------------------------------------------------------------------------------------------------
for symbol in symbols:

    # Enter the directory you want to search in
    my_dir = 'code/data/historic_data'

    # Enter the file you want to search for
    file_name = symbol+'.csv'

    # Check if the file exists
    if os.path.isfile(os.path.join(my_dir, file_name)):
        continue
    else:
        print(symbol)
