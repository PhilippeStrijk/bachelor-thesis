# Import the required modules 
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
import pandas as pd
# Import the required modules 
import webbrowser

# Open the URL


# Read the csv into a pandas dataframe
df = pd.read_csv('code/data/data/constituents_csv.csv')

symbols = df['Symbol']

print(symbols)

for symbol in symbols:
    webbrowser.open('https://query1.finance.yahoo.com/v7/finance/download/' + symbol + '?period1=1232841600&period2=1671926400&interval=1d&events=history&includeAdjustedClose=true')
