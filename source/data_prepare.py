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
    N = len(df)
    y = []
    sample = []
    y.append([0,1,0])
    dic = {}
    for i in range(N-1):
      if df[df.columns[0]][i] > df[df.columns[0]][i+1]:
          y.append([0,0,1]) # baixa
      if df[df.columns[0]][i] < df[df.columns[0]][i+1]:
          y.append([1,0,0]) # alta
      if df[df.columns[0]][i] == df[df.columns[0]][i+1]:
          y.append([0,1,0]) # estável
    dic['df'] = df
    dic['y'] = y
    return dic

  def init_samples(self,df):
    return np.zeros(len(df), dtype=[('input',  float, len(df.columns)), ('output', list, 1.0)])
  
  def data_frame_row_to_array(self, df):
    return [list(x) for x in df.to_records(index=False)]

  def prepare_data(self):
    """
      formado de saída do dataframe deve ser: [(x1, ..., x_n), y]
      do tipo numpy.array(ROW_SIZE, dtype=[('input',  float, X_N), ('output', float, Y)])
      a = np.zeros(len(df), dtype=[('input',  float, len(df.columns)), ('output', float, 1.0)])
      k = [list(x) for x in df.to_records(index=False)]
      f = np.asarray(k[0])
      np.copyto(a[0][0],f)
    """
    s_data = self.sliced_data()
    df = self.y_data(s_data)
    a = self.init_samples(df['df'])
    k = self.data_frame_row_to_array(df['df'])
    for i in range(len(a)):
      x = np.asarray(k[i])
      y = df['y'][i]
      if isinstance(a[i][0], np.ndarray):
        np.copyto(a[i][0],x)
      else:
        a[i][0] = x
      a[i][1] = y
      print
    return a

# -----------------------------------------------------------------------------
if __name__ == '__main__':
  file = '../data/all_closes_percentage.csv'
  st1 = 'americas_bvsp'
  st2 = 'americas_gsptse'
  st3 = ['americas_ipsa',"americas_merv","americas_mxx","asia_000001ss","asia_aord","asia_axjo","asia_bsesn","asia_hsi","asia_jkse"]
  stock = [st1,st2] + st3
  DP1 = DataPrepare(file)
  sample = DP1.prepare_data()
  print sample





