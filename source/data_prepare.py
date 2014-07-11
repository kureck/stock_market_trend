import pandas as pd
import numpy as np

def read_file(file):
  return pd.read_csv(file)

def prepare_data(df):
  """
    formado de saída do dataframe deve ser: [(x1, ..., x_n), y]
    do tipo numpy.array(ROW_SIZE, dtype=[('input',  float, X_N), ('output', float, Y)])
  """ 
  df['y'] = np.ones(np.shape(len(df)))
  N = len(df)
  sample = []
  for i in range(N-1):
    if df['americas_bvsp'][i] > df['americas_bvsp'][i+1]:
        df['y'][i+1] = -1
    if df['americas_bvsp'][i] < df['americas_bvsp'][i+1]:
        df['y'][i+1] = 1
    if df['americas_bvsp'][i] == df['americas_bvsp'][i+1]:
        df['y'][i+1] = 0.5
  # x = pd.DataFrame()
  # x['Date'] = df['Date']
  # x['Close'] = df['Close']
  x = np.vstack(df['americas_bvsp'])
  y = np.array(df['y'])
  return x,y

def data(file):
  df = read_file(file)
  return prepare_data(df)



