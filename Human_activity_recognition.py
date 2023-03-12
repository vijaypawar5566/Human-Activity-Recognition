# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Human Activity Recognition Using SmartDevice Data(Phone/Watch)

# COMMAND ----------

# DBTITLE 1,Reading accel data of user 10

# The applied options are for CSV files. For other file types, these will be ignored.
col = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z']
raw_par_10_phone_accel = spark.read.format("csv") \
             .option("header", "false") \
             .option("inferSchema", "true") \
             .option("delimiter", ",") \
             .load("s3://humanactivity/wisdm-dataset/raw/phone/accel/data_1610_accel_phone.txt") \
             .toDF(*col)

display(raw_par_10_phone_accel)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## EDA

# COMMAND ----------

raw_par_10_phone_accel.show(5)

# COMMAND ----------

raw_par_10_phone_accel.dtypes

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
raw_par_10_phone_accel=raw_par_10_phone_accel.withColumn('z',regexp_replace('z',';',''))

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType


# COMMAND ----------

from pyspark.sql.types import DoubleType
raw_par_10_phone_accel = raw_par_10_phone_accel.withColumn("z", col("z").cast(DoubleType()))

# COMMAND ----------

raw_par_10_phone_accel.dtypes

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp
raw_par_10_phone_accel = raw_par_10_phone_accel.withColumn('timestamp', col('timestamp').cast('timestamp'))

# COMMAND ----------

raw_par_10_phone_accel.dtypes

# COMMAND ----------

raw_par_10_phone_accel.show(5)

# COMMAND ----------

# DBTITLE 1,Null value check
from pyspark.sql.functions import *
columns = ['participant_id','activity_code','timestamp', 'x','y','z']
for i in columns:
    print(i,raw_par_10_phone_accel.filter(raw_par_10_phone_accel[i].isNull()).count())

# COMMAND ----------

# DBTITLE 1,Dictionary to map activity_code column
activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

# COMMAND ----------

def activity_codes_mapping_udf(code):
    return activity_codes_mapping.get(code, 'unknown')

# COMMAND ----------

# DBTITLE 1,Colour code for each Activity
activity_color_map = {activity_codes_mapping['A']: 'lime',
                      activity_codes_mapping['B']: 'red',
                      activity_codes_mapping['C']: 'blue',
                      activity_codes_mapping['D']: 'orange',
                      activity_codes_mapping['E']: 'yellow',
                      activity_codes_mapping['F']: 'lightgreen',
                      activity_codes_mapping['G']: 'greenyellow',
                      activity_codes_mapping['H']: 'magenta',
                      activity_codes_mapping['I']: 'gold',
                      activity_codes_mapping['J']: 'cyan',
                      activity_codes_mapping['K']: 'purple',
                      activity_codes_mapping['L']: 'lightgreen',
                      activity_codes_mapping['M']: 'violet',
                      activity_codes_mapping['O']: 'limegreen',
                      activity_codes_mapping['P']: 'deepskyblue',   
                      activity_codes_mapping['Q']: 'mediumspringgreen',
                      activity_codes_mapping['R']: 'plum',
                      activity_codes_mapping['S']: 'olive'}

# COMMAND ----------

# DBTITLE 1,Histogram
column_data=raw_par_10_phone_accel.select('x')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

column_data=raw_par_10_phone_accel.select('y')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

column_data=raw_par_10_phone_accel.select('z')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

# DBTITLE 1,Fun to get Line chart of accel data per activity
def show_accel_per_activity(device, df, act, interval_in_sec = None):
  ''' Plots acceleration time history per activity '''

  df1 = df.loc[df.activity_code == act].copy()
  df1.reset_index(drop = True, inplace = True)
  
  df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds
  
  if interval_in_sec == None:
    ax = df1[:].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)
  else:
    ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)

  ax.set_xlabel('duration  (sec)', fontsize = 15)
  ax.set_ylabel('acceleration  (m/sec^2)',fontsize = 15)
  ax.set_title('Acceleration:   Device: ' + device + '      Activity:  ' +activity_codes_mapping[act], fontsize = 15)

# COMMAND ----------

raw_par_10_phone_accel_pan=raw_par_10_phone_accel.toPandas()

# COMMAND ----------

for key in activity_codes_mapping:
  show_accel_per_activity('Phone', raw_par_10_phone_accel_pan, key, 10)

# COMMAND ----------

# DBTITLE 1,Reading accel data of user 20
col = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z']
raw_par_20_watch_accel = spark.read.format("csv") \
             .option("header", "false") \
             .option("inferSchema", "true") \
             .option("delimiter", ",") \
             .load("s3://humanactivity/wisdm-dataset/raw/watch/accel/data_1620_accel_watch.txt") \
             .toDF(*col)

display(raw_par_20_watch_accel)

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
raw_par_20_watch_accel=raw_par_20_watch_accel.withColumn('z',regexp_replace('z',';',''))


# COMMAND ----------

from pyspark.sql.types import DoubleType
raw_par_20_watch_accel = raw_par_20_watch_accel.withColumn("z", col("z").cast(DoubleType()))

# COMMAND ----------

raw_par_20_watch_accel.dtypes

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp
raw_par_20_watch_accel = raw_par_20_watch_accel.withColumn('timestamp', col('timestamp').cast('timestamp'))

# COMMAND ----------

raw_par_20_watch_accel.show(4)

# COMMAND ----------

raw_par_20_watch_accel.dtypes

# COMMAND ----------

from pyspark.sql.functions import *
columns = ['participant_id','activity_code','timestamp', 'x','y','z']
for i in columns:
    print(i,raw_par_20_watch_accel.filter(raw_par_20_watch_accel[i].isNull()).count())

# COMMAND ----------

# DBTITLE 1,Histogram
column_data=raw_par_20_watch_accel.select('x')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

column_data=raw_par_20_watch_accel.select('y')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

column_data=raw_par_20_watch_accel.select('z')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

raw_par_20_watch_accel_pan=raw_par_20_watch_accel.toPandas()

# COMMAND ----------

for key in activity_codes_mapping:
  show_accel_per_activity('Watch', raw_par_20_watch_accel_pan, key, 10)

# COMMAND ----------

# DBTITLE 1,Reading gyro data of user 35
col = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z']
raw_par_35_phone_gyro = spark.read.format("csv") \
             .option("header", "false") \
             .option("inferSchema", "true") \
             .option("delimiter", ",") \
             .load("s3://humanactivity/wisdm-dataset/raw/phone/gyro/data_1635_gyro_phone.txt") \
             .toDF(*col)

display(raw_par_35_phone_gyro)

# COMMAND ----------

raw_par_35_phone_gyro=raw_par_35_phone_gyro.withColumn('z',regexp_replace('z',';',''))


# COMMAND ----------

from pyspark.sql.functions import col
raw_par_35_phone_gyro = raw_par_35_phone_gyro.withColumn("z", col("z").cast(DoubleType()))
raw_par_35_phone_gyro.dtypes


# COMMAND ----------

# DBTITLE 1,Null Value Check
from pyspark.sql.functions import *
columns = ['participant_id','activity_code','timestamp', 'x','y','z']
for i in columns:
    print(i,raw_par_35_phone_gyro.filter(raw_par_35_phone_gyro[i].isNull()).count())


# COMMAND ----------

# DBTITLE 1,Histogram
column_data=raw_par_35_phone_gyro.select('x')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()


column_data=raw_par_35_phone_gyro.select('y')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()



column_data=raw_par_35_phone_gyro.select('z')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

raw_par_35_phone_gyro_pan=raw_par_35_phone_gyro.toPandas()

# COMMAND ----------

# DBTITLE 1,Fun to get line chart of angular velocity per activity
def show_ang_velocity_per_activity(device, df, act, interval_in_sec = None):
  ''' Plots angular volocity time history per activity '''

  df1 = df.loc[df.activity_code == act].copy()
  df1.reset_index(drop = True, inplace = True)

  df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds

  if interval_in_sec == None:
    ax = df1[:].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)
  else:
    ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)

  ax.set_xlabel('duration  (sec)', fontsize = 15)
  ax.set_ylabel('angular velocity  (rad/sec)',fontsize = 15)
  ax.set_title('Angular velocity:  Device: ' + device + '      Activity:  ' +activity_codes_mapping[act] , fontsize = 15)

# COMMAND ----------

for key in activity_codes_mapping:
  show_ang_velocity_per_activity('Phone', raw_par_35_phone_gyro_pan, key)

# COMMAND ----------

col = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z']
raw_par_35_watch_gyro = spark.read.format("csv") \
             .option("header", "false") \
             .option("inferSchema", "true") \
             .option("delimiter", ",") \
             .load("s3://humanactivity/wisdm-dataset/raw/watch/gyro/data_1635_gyro_watch.txt") \
             .toDF(*col)

display(raw_par_35_watch_gyro)

# COMMAND ----------

raw_par_35_watch_gyro=raw_par_35_watch_gyro.withColumn('z',regexp_replace('z',';',''))

# COMMAND ----------

from pyspark.sql.functions import col
raw_par_35_watch_gyro = raw_par_35_watch_gyro.withColumn("z", col("z").cast(DoubleType()))
raw_par_35_watch_gyro.dtypes

# COMMAND ----------

# DBTITLE 1,Null Value Check
from pyspark.sql.functions import *
columns = ['participant_id','activity_code','timestamp', 'x','y','z']
for i in columns:
    print(i,raw_par_35_watch_gyro.filter(raw_par_35_watch_gyro[i].isNull()).count())

# COMMAND ----------

# DBTITLE 1,Histogram
column_data=raw_par_35_watch_gyro.select('x')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()


column_data=raw_par_35_watch_gyro.select('y')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()



column_data=raw_par_35_watch_gyro.select('z')
histogram = column_data.rdd.flatMap(lambda x: x).histogram(4)

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(column_data.rdd.flatMap(lambda x: x).collect(), bins=50, color='green')
plt.show()

# COMMAND ----------

raw_par_35_watch_gyro_pan=raw_par_35_watch_gyro.toPandas()

# COMMAND ----------

# DBTITLE 1,Line-Charts By Activity
for key in activity_codes_mapping:
  show_ang_velocity_per_activity('Watch', raw_par_35_watch_gyro_pan, key)

# COMMAND ----------

features = ['ACTIVITY',
            'X0', # 1st bin fraction of x axis acceleration distribution
            'X1', # 2nd bin fraction ...
            'X2',
            'X3',
            'X4',
            'X5',
            'X6',
            'X7',
            'X8',
            'X9',
            'Y0', # 1st bin fraction of y axis acceleration distribution
            'Y1', # 2nd bin fraction ...
            'Y2',
            'Y3',
            'Y4',
            'Y5',
            'Y6',
            'Y7',
            'Y8',
            'Y9',
            'Z0', # 1st bin fraction of z axis acceleration distribution
            'Z1', # 2nd bin fraction ...
            'Z2',
            'Z3',
            'Z4',
            'Z5',
            'Z6',
            'Z7',
            'Z8',
            'Z9',
            'XAVG', # average sensor value over the window (per axis)
            'YAVG',
            'ZAVG',
            'XPEAK', # Time in milliseconds between the peaks in the wave associated with most activities. heuristically determined (per axis)
            'YPEAK',
            'ZPEAK',
            'XABSOLDEV', # Average absolute difference between the each of the 200 readings and the mean of those values (per axis)
            'YABSOLDEV',
            'ZABSOLDEV',
            'XSTANDDEV', # Standard deviation of the 200 window's values (per axis)  ***BUG!***
            'YSTANDDEV',
            'ZSTANDDEV',
            'XVAR', # Variance of the 200 window's values (per axis)   ***BUG!***
            'YVAR',
            'ZVAR',
            'XMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'XMFCC1',
            'XMFCC2',
            'XMFCC3',
            'XMFCC4',
            'XMFCC5',
            'XMFCC6',
            'XMFCC7',
            'XMFCC8',
            'XMFCC9',
            'XMFCC10',
            'XMFCC11',
            'XMFCC12',
            'YMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'YMFCC1',
            'YMFCC2',
            'YMFCC3',
            'YMFCC4',
            'YMFCC5',
            'YMFCC6',
            'YMFCC7',
            'YMFCC8',
            'YMFCC9',
            'YMFCC10',
            'YMFCC11',
            'YMFCC12',
            'ZMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'ZMFCC1',
            'ZMFCC2',
            'ZMFCC3',
            'ZMFCC4',
            'ZMFCC5',
            'ZMFCC6',
            'ZMFCC7',
            'ZMFCC8',
            'ZMFCC9',
            'ZMFCC10',
            'ZMFCC11',
            'ZMFCC12',
            'XYCOS', # The cosine distances between sensor values for pairs of axes (three pairs of axes)
            'XZCOS',
            'YZCOS',
            'XYCOR', # The correlation between sensor values for pairs of axes (three pairs of axes)
            'XZCOR',
            'YZCOR',
            'RESULTANT', # Average resultant value, computed by squaring each matching x, y, and z value, summing them, taking the square root, and then averaging these values over the 200 readings
            'PARTICIPANT'] # Categirical: 1600 -1650

len(features)

# COMMAND ----------

# DBTITLE 1,Feature Dataset
path = 's3://humanactivity/wisdm-dataset/arff_files/phone/all_phone_accel'
all_phone_accel = spark.read.csv(path,header=True)
all_phone_accel.select("X0").show(4)


# COMMAND ----------

all_phone_accel.columns

# COMMAND ----------

all_phone_accel.dtypes

# COMMAND ----------

# DBTITLE 1,Casting each column except activity into DoubleType()
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
for coli in all_phone_accel.columns:
    if coli == 'ACTIVITY':
        continue
    else:
        all_phone_accel = all_phone_accel.withColumn(coli, col(coli).cast(DoubleType()))
        

# COMMAND ----------

all_phone_accel.dtypes

# COMMAND ----------

# DBTITLE 1,Null Value Check
for coli in all_phone_accel.columns:
    print(coli,all_phone_accel.filter(all_phone_accel[coli].isNull()).count())

# COMMAND ----------

# DBTITLE 1,Row count per Activity
import matplotlib.pyplot as plt
import pandas as pd

# Group by activity and count the rows
activity_counts = all_phone_accel.groupBy('ACTIVITY') \
    .count() \
    .orderBy('count', ascending=False)

# Convert to Pandas DataFrame
activity_counts_pd = activity_counts.toPandas()

# Plot the results
_ = activity_counts_pd.plot(kind='bar', x='ACTIVITY', y='count',
                            figsize=(15, 5), color='purple',
                            title='row count per activity',
                            legend=True, fontsize=15)
plt.show()

# COMMAND ----------

# DBTITLE 1,Row count per Participant
# Group by activity and count the rows
activity_counts = all_phone_accel.groupBy('PARTICIPANT') \
    .count() \
    .orderBy('count', ascending=False)

# Convert to Pandas DataFrame
activity_counts_pd = activity_counts.toPandas()

# Plot the results
_ = activity_counts_pd.plot(kind='bar', x='PARTICIPANT', y='count',
                            figsize=(15, 5), color='purple',
                            title='row count per PARTICIPANT',
                            legend=True, fontsize=15)
plt.show()

# COMMAND ----------

all_phone_accel[['XABSOLDEV', 'YABSOLDEV','ZABSOLDEV','XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR']].show(4)

# COMMAND ----------

# DBTITLE 1,Droping Erroneous Columns
all_phone_accel = all_phone_accel.drop('XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR')

# COMMAND ----------

all_phone_accel.columns

# COMMAND ----------

# DBTITLE 1,Preprocessing
all_phone_accel = all_phone_accel.drop('PARTICIPANT')

# COMMAND ----------

all_phone_accel.columns

# COMMAND ----------

# DBTITLE 1,Train test data
X_train = spark.read.csv("s3://humanactivity/Train_test_spllit/X_train",header=True)
X_test = spark.read.csv("s3://humanactivity/Train_test_spllit/X_test",header=True)
y_train=spark.read.csv("s3://humanactivity/Train_test_spllit/y_train",header=True)
y_test= spark.read.csv("s3://humanactivity/Train_test_spllit/y_test",header=True)

# COMMAND ----------

X_train.show(4)

# COMMAND ----------

X_test.show(4)

# COMMAND ----------

y_train.show(4)

# COMMAND ----------

y_test.show(4)

# COMMAND ----------

X_test.count(),X_train.count()

# COMMAND ----------

y_test.count(),y_train.count()

# COMMAND ----------

# DBTITLE 1,Cluster-Diagrame
par_23_df = spark.read.csv("s3://humanactivity/Cluster_diagram/par_23_df",header=True)

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Convert the PySpark DataFrame to a Pandas DataFrame
par_23_pd_df = par_23_df.toPandas()

# Extract the features and the target variable from the Pandas DataFrame
yy = par_23_pd_df['ACTIVITY']
XX = par_23_pd_df.drop(['ACTIVITY','PARTICIPANT','ACT','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR'], axis = 1)

# Apply t-SNE to reduce the dimensionality of the feature space to 2 dimensions
tsne = TSNE(n_components=2, random_state=300)
X_2d = tsne.fit_transform(XX)

# Plot the t-SNE visualization using Matplotlib
target_ids = tuple(activity_codes_mapping.keys())

plt.figure(figsize=(10, 10))
colors = 'lime', 'red', 'blue', 'orange', 'yellow', 'lightgreen', 'greenyellow', 'magenta', 'gold', 'cyan', 'purple', 'lightgreen', 'violet', 'limegreen', 'deepskyblue', 'mediumspringgreen', 'plum', 'olive'

for i, c, label in zip(target_ids, colors, tuple(activity_codes_mapping.values())):
    plt.scatter(X_2d[yy == i, 0], X_2d[yy == i, 1], c=c, label=label)

plt.legend()
plt.show()

# COMMAND ----------

par_35_df = spark.read.csv("s3://humanactivity/Cluster_diagram/par_35_df",header=True)

# COMMAND ----------

# Convert the PySpark DataFrame to a Pandas DataFrame
par_35_pd_df = par_35_df.toPandas()

# Extract the features and the target variable from the Pandas DataFrame
yy = par_23_pd_df['ACTIVITY']
XX = par_23_pd_df.drop(['ACTIVITY','PARTICIPANT','ACT','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR'], axis = 1)

# Apply t-SNE to reduce the dimensionality of the feature space to 2 dimensions
tsne = TSNE(n_components=2, random_state=300)
X_2d = tsne.fit_transform(XX)

# Plot the t-SNE visualization using Matplotlib
target_ids = tuple(activity_codes_mapping.keys())

plt.figure(figsize=(10, 10))
colors = 'lime', 'red', 'blue', 'orange', 'yellow', 'lightgreen', 'greenyellow', 'magenta', 'gold', 'cyan', 'purple', 'lightgreen', 'violet', 'limegreen', 'deepskyblue', 'mediumspringgreen', 'plum', 'olive'

for i, c, label in zip(target_ids, colors, tuple(activity_codes_mapping.values())):
    plt.scatter(X_2d[yy == i, 0], X_2d[yy == i, 1], c=c, label=label)

plt.legend()
plt.show()

# COMMAND ----------

par_40_df = spark.read.csv("s3://humanactivity/Cluster_diagram/par_40_df",header=True)

# COMMAND ----------

# Convert the PySpark DataFrame to a Pandas DataFrame
par_40_pd_df = par_40_df.toPandas()

# Extract the features and the target variable from the Pandas DataFrame
yy = par_23_pd_df['ACTIVITY']
XX = par_23_pd_df.drop(['ACTIVITY','PARTICIPANT','ACT','XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR'], axis = 1)

# Apply t-SNE to reduce the dimensionality of the feature space to 2 dimensions
tsne = TSNE(n_components=2, random_state=300)
X_2d = tsne.fit_transform(XX)

# Plot the t-SNE visualization using Matplotlib
target_ids = tuple(activity_codes_mapping.keys())

plt.figure(figsize=(10, 10))
colors = 'lime', 'red', 'blue', 'orange', 'yellow', 'lightgreen', 'greenyellow', 'magenta', 'gold', 'cyan', 'purple', 'lightgreen', 'violet', 'limegreen', 'deepskyblue', 'mediumspringgreen', 'plum', 'olive'

for i, c, label in zip(target_ids, colors, tuple(activity_codes_mapping.values())):
    plt.scatter(X_2d[yy == i, 0], X_2d[yy == i, 1], c=c, label=label)

plt.legend()
plt.show()

# COMMAND ----------

# DBTITLE 1,Application of Machine Learning Classification Models:
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import count

counts = y_train.groupBy('Y').agg(count('*').alias('count'))
activity_counts_pd = counts.toPandas()

activity_counts_pd.plot(kind='bar', x='Y', y='count', color='red', figsize=(15,5), legend=False, fontsize=15)
plt.title('Row count per activity', fontsize=15)
plt.show()


# COMMAND ----------

# DBTITLE 1,## y_test Distribution
counts = y_test.groupBy('ACTIVITY').agg(count('*').alias('count'))
activity_counts_pd = counts.toPandas()

activity_counts_pd.plot(kind='bar', x='ACTIVITY', y='count', color='red', figsize=(15,5), legend=False, fontsize=15)
plt.title('Row count per activity', fontsize=15)
plt.show()

# COMMAND ----------

# DBTITLE 1,K Nearest Neighbors Model:
from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

knn_classifier = KNeighborsClassifier()

# COMMAND ----------

my_param_grid = {'n_neighbors': [5, 10, 20], 'leaf_size': [20, 30, 40]}

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold

my_cv = StratifiedShuffleSplit(n_splits=5, train_size=0.7, test_size=0.3)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
knn_model_gs = GridSearchCV(estimator = knn_classifier, 
                            param_grid = my_param_grid,
                            cv = my_cv, 
                            scoring ='accuracy')

# COMMAND ----------

X_train_pan = X_train.toPandas()
X_train_pan.shape

# COMMAND ----------

y_train_pan = y_train.toPandas()
y_train_pan.shape

# COMMAND ----------

knn_model_gs.fit(X_train_pan, y_train_pan)

# COMMAND ----------

knn_best_classifier = knn_model_gs.best_estimator_
knn_best_classifier

# COMMAND ----------

print(knn_model_gs.best_params_)

# COMMAND ----------

knn_model_gs.cv_results_

# COMMAND ----------

knn_best_classifier.get_params()

# COMMAND ----------

scores = cross_val_score(knn_best_classifier, X_train_pan, y_train_pan, cv=my_cv, scoring='accuracy')
list(scores)

# COMMAND ----------

y_train_pred=knn_best_classifier.predict(X_train_pan)

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_train_pan, y_pred=y_train_pred)

# COMMAND ----------

X_test_pan = X_test.toPandas()
y_test_pred = knn_best_classifier.predict(X_test_pan)

# COMMAND ----------

y_test_pan = y_test.toPandas()

# COMMAND ----------

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test_pan,
                      y_pred=y_test_pred)
    
cm_act = pd.DataFrame(cm,
                      index = knn_best_classifier.classes_,
                      columns = knn_best_classifier.classes_)

cm_act.columns = activity_codes_mapping.values()
cm_act.index = activity_codes_mapping.values()
cm_act

# COMMAND ----------

import seaborn as sns
sns.set(font_scale=1.6)
fig, ax = plt.subplots(figsize=(12,10))
_ = sns.heatmap(cm_act, cmap="YlGnBu")

# COMMAND ----------

import numpy as np
accuracy_per_activity = pd.DataFrame([cm_act.iloc[i][i]/np.sum(cm_act.iloc[i]) for i in range(18)],index=activity_codes_mapping.values())
accuracy_per_activity

# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test_pan,
                            y_pred=y_test_pred))

# COMMAND ----------

accuracy_score(y_true = y_test_pan, y_pred = y_test_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Models mean prediction accuracy scores summary: 
# MAGIC 
# MAGIC Decision Tree: 0.33
# MAGIC 
# MAGIC Random Forest: 0.44
# MAGIC 
# MAGIC Logistic Regression: 0.38
# MAGIC 
# MAGIC ## KNN: 0.75

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Insights & Conclusions:**
# MAGIC 
# MAGIC 1) The K Nearest Neighbor classifier has prooved to provide substantial higher prediction accuracy than the rest of the models (overall mean accuracy ~0.75 on test set) in this case.
# MAGIC 
# MAGIC 2) The analysis demonstrated typical differentiation of detection accuracy accross the various physical activities

# COMMAND ----------


