import pandas as pd
import numpy as np

def read_file(file):
  return pd.read_csv(file, parse_dates=['Date'])

def prepare_data(df):
  df['y'] = np.zeros(np.shape(len(df)))
  N = len(df)
  for i in range(N-1):
    if df['Close'][i] > df['Close'][i+1]:
        df['y'][i] = 1
    if df['Close'][i] < df['Close'][i+1]:
        df['y'][i] = -1
    if df['Close'][i] == df['Close'][i+1]:
        df['y'][i] = 0.5
  # x = pd.DataFrame()
  # x['Date'] = df['Date']
  # x['Close'] = df['Close']
  x = np.vstack(df['Close'])
  # x = np.array([P_x() for i in range(N)])
  y = np.array(df['y'])
  return x,y

def data(file):
  df = read_file('../data/IBOVESPA.csv')
  return prepare_data(df)



