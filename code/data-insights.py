# Import the required modules 
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd

# Create a spark session 
spark = SparkSession.builder.appName('csv_reader').getOrCreate()

# Create a dataframe reader
df_reader = DataFrameReader(spark)

# Read the csv file into a spark dataframe
df = df_reader.csv('code/data/s&p500_historic_data.csv', header=True)
df = df.withColumnRenamed("_c0", 'Index')

# Make sure the dataframe has the correct types
df = df.withColumn("Open", df["Open"].cast("float"))
df = df.withColumn("High", df["High"].cast("float"))
df = df.withColumn("Low", df["Low"].cast("float"))
df = df.withColumn("Close", df["Close"].cast("float"))
df = df.withColumn("Adj Close", df["Adj Close"].cast("float"))
df = df.withColumn("Volume", df["Volume"].cast("float"))
df = df.withColumn("Date", F.to_date(df["Date"]))

# Use the constituents .csv to iterate through the different stocks
stocks = pd.read_csv('code/data/data/constituents_csv.csv')
stocks = stocks['Symbol']

top_stocks = pd.DataFrame()

for stock in stocks:
    analyze_df = df.filter(df['Symbol'] == stock).select('*')
    analyze_df.select(['Close','Date']).toPandas().plot(x='Date', y='Close').set_title(stock)

    plt.show()

df.show()

spark.stop()