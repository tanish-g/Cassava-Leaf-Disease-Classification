class split():
  def __init__(self,df):
      self.df=df
  def get_train_val_split(self):
      #Remove Duplicates
      train_df=self.df[self.df['fold']!=0].reset_index(drop=True)
      valid_df=self.df[self.df['fold']==0].reset_index(drop=True)
      return train_df,valid_df
