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
    self.samples = self.data()


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
          df['y'][i+1] = y = -1
      if df[df.columns[0]][i] < df[df.columns[0]][i+1]:
          df['y'][i+1] = y = 1
      if df[df.columns[0]][i] == df[df.columns[0]][i+1]:
          df['y'][i+1] = y = 0.5
    return df

  def init_samples(self,df):
    return np.zeros(len(df), dtype=[('input',  float, len(df.columns)), ('output', float, 1.0)])
  
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
    a = self.init_samples(df)
    k = [list(x) for x in df.to_records(index=False)]
    return a

  def data(self):
    pass

# -----------------------------------------------------------------------------
if __name__ == '__main__':
  file = '../data/all_closes_percentage.csv'
  st1 = 'americas_bvsp'
  st2 = 'americas_gsptse'
  st3 = 'americas_ipsa'
  DP1 = DataPrepare(file, [st1, st2, st3])
  s_data = DP1.sliced_data()
  y_data = DP1.y_data(s_data)
  a = DP1.init_samples(y_data)
  print a





