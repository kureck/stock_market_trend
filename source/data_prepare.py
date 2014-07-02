import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

def read_file(file):
  return pd.read_csv(file, parse_dates=['Date'], index_col='Date')

def prepare_data(df):
  df['y'] = np.zeros(np.shape(len(df)))
  N = len(df)
  for i in range(N-1):
    if df['Close'][i] > df['Close'][i+1]:
        df['y'][i] = 1
    if df['Close'][i] < df['Close'][i+1]:
        df['y'][i] = 0
    if df['Close'][i] == df['Close'][i+1]:
        df['y'][i] = 0.5
  return df

df = read_file('../data/IBOVESPA.csv')
d = prepare_data(df)



