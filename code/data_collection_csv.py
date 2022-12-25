# Import the required modules 
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
import pandas as pd
import os
import webbrowser
import numpy as np
# Read the csv into a pandas dataframe
df = pd.read_csv('code/data/data/constituents_csv.csv')
symbols = df['Symbol']

# ---------------------------- BE CAREFUL, WILL DOWNLOAD A BUNCH OF CSV FILES, BROWSER MAY CRASH ----------------------------------------
""" for symbol in symbols:
    webbrowser.open('https://query1.finance.yahoo.com/v7/finance/download/' + symbol + '?period1=1232841600&period2=1671926400&interval=1d&events=history&includeAdjustedClose=true')
 """
# ----------------------------------------------------------------------------------------------------------------------------------------
index = -1
for symbol in symbols:
    index += 1
    # Enter the directory you want to search in
    my_dir = 'code/data/historic_data'

    # Enter the file you want to search for
    file_name = symbol+'.csv'
    # Check if the file exists
    if os.path.isfile(os.path.join(my_dir, file_name)):
        continue
    else:
        df.drop(index, inplace = True)
        continue

print(df)

symbols = df['Symbol']

df['Date'] = np.nan
df['Open'] = np.nan
df['High'] = np.nan
df['Low'] = np.nan
df['Close'] = np.nan
df['Adj Close'] = np.nan
df['Volume'] = np.nan

print(df)
final_df = pd.DataFrame()
index = 0
for row in df.values:
    index += 1
    print(index)
    symbol = row[0]
    name = row[1]
    sector = row[2]
    historic_data = pd.read_csv("code/data/historic_data/" + symbol + ".csv")
    new_df = pd.DataFrame()
    new_df['Symbol'] = np.nan
    new_df['Name'] = np.nan
    new_df['Sector'] = np.nan

    new_df = new_df.append(historic_data)

    new_df['Symbol'] = symbol
    new_df['Name'] = name
    new_df['Sector'] = sector
    final_df = pd.concat([final_df, new_df])

final_df.to_csv('code/data/s&p500_historic_data.csv')