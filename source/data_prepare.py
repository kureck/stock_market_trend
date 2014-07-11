#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class DataPrepare:
  """DataPrepare Class
    
    Class to prepare data to feed the MLP
  """

  def __init__(self, file, args=['americas_bvsp']):
    """
      :type file: string
      :param file: dataset file name

      :type args: list of strings
      :param args: a list of stock names
    """
    self.headers = args
    self.df = self.read_file(file)

  def read_file(self, file):
    return pd.read_csv(file)

  def sliced_data(self):
    return self.df[self.headers]
  
  def y_data(self, df):
    df['y'] = np.ones(np.shape(len(df)))
    N = len(df)
    sample = []
    y = 1.0
    for i in range(N-1):
      if df[df.columns[0]][i] > df[df.columns[0]][i+1]:
          df['y'][i+1] = y = 0
      if df[df.columns[0]][i] < df[df.columns[0]][i+1]:
          df['y'][i+1] = y = 1
      if df[df.columns[0]][i] == df[df.columns[0]][i+1]:
          df['y'][i+1] = y = 0.5
    return df

  def init_samples(self,df):
    return np.zeros(len(df), dtype=[('input',  float, len(df.columns)-1), ('output', float, 1.0)])
  
  def data_frame_row_to_array(self, df):
    return [list(x) for x in df.to_records(index=False)]

  def prepare_data(self, df):
    """
      formado de saída do dataframe deve ser: [(x1, ..., x_n), y]
      do tipo numpy.array(ROW_SIZE, dtype=[('input',  float, X_N), ('output', float, Y)])
      a = np.zeros(len(df), dtype=[('input',  float, len(df.columns)), ('output', float, 1.0)])
      k = [list(x) for x in df.to_records(index=False)]
      f = np.asarray(k[0])
      np.copyto(a[0][0],f)
    """
    df = self.y_data(df)
    print df
    raw_input()
    a = self.init_samples(df)
    print a
    raw_input()
    k = self.data_frame_row_to_array(df)
    print k
    raw_input()
    for i in range(len(a)):
      x = np.asarray(k[i][0:-1])
      y = k[i][-1]
      np.copyto(a[i][0],x)
      a[i][1] = y
    return a

# -----------------------------------------------------------------------------
if __name__ == '__main__':
  file = '../data/all_closes_percentage.csv'
  st1 = 'americas_bvsp'
  st2 = 'americas_gsptse'
  st3 = 'americas_ipsa'
  DP1 = DataPrepare(file, [st1, st2])
  s_data = DP1.sliced_data()
  y_data = DP1.y_data(s_data)
  sample = DP1.prepare_data(y_data)
  print sample





