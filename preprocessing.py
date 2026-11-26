import numpy as np
import pandas as pd
def encode_CB(col1,col2,df1=None,df2=None):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 

def apply_com(df):
   df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')
   df['card1_addr1'] = (df['card1'].astype(str)+'_'+df['addr1'].astype(str))
   df['card1_addr1_P_emaildomain'] = (df['card1_addr1'].astype(str)+'_'+df['P_emaildomain'].astype(str))
   return df


def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')

def apply_freq(df, freq_maps):
   freq_cols = ['addr1','card1','card2','card3','P_emaildomain','card1_addr1','card1_addr1_P_emaildomain']
   for col in freq_cols:
      new_col_name = col+'_FE'
      df[new_col_name] = (df[col].map(freq_maps[col]).astype('float32'))
      #df[new_col_name] = df[new_col_name].fillna(-1)
   return df

def encode_AG(main_columns, uids, aggregations=['mean'], train_df=None, test_df=None, 
              fillna=True, usena=False):
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)

def apply_AG(df, agg_maps):
    for group_col, data in agg_maps.items():
        indices = df[group_col].map(data['key_to_idx'])

        valid = indices.notna().to_numpy()
        index_values = indices.fillna(0).to_numpy(dtype=np.int64)

        for j, (main_col, agg_type) in enumerate(data['columns']):
            new_col_name = f'{main_col}_{group_col}_{agg_type}'

            output = np.full(len(df), np.nan, dtype=np.float32)
            output[valid] = data['values'][index_values[valid], j]

            df[new_col_name] = output

    return df

def apply_le(df, encoders):
    for col, encoder in encoders.items():
        df[col] = df[col].apply(
            lambda x: np.nan if pd.isna(x) else str(x)
        )

        known_values = set(encoder.classes_)
        known_mask = df[col].isin(known_values)

        encoded = pd.Series(np.nan, index=df.index)

        encoded.loc[known_mask] = encoder.transform(
            df.loc[known_mask, col]
        )

        df[col] = encoded

    return df

ftnl=['V338', 'V331', 'V333', 'V334', 'V327', 'V328', 'V329', 'V330', 'V14', 'V304', 'V303', 'V302', 'V305', 'V8', 
                                  'V46', 'V50', 'V227', 'V92', 'V174', 'V153', 'V227', 'V223', 'V231', 'V232', 'V235', 'V237', 'V205', 'V228', 'dist2',
                                 'V244', 'V241', 'V242', 'V239', 'V249', 'V27', 'V238', 'V65', 'V260', 'V255', 'V252', 'V31', 'V32', 'V250',
                                 'V28', 'V246', 'V71', 'V299', 'V297', 'V295', 'V284', 'V68', 'V324', 'V325', 'V17', 'V319', 'V1', 'V18', 'V15', 'V16', 'V277', 'V185', 'V181', 'V182', 'V192', 'V193', 'V194', 'V196', 'V186', 'V122', 'V103', 'V107',
 'V106', 'V339', 'V88', 'V89', 'V98', 'V104', 'V118', 'V137', 'V41', 'V177', 'V173', 'V171', 'V168', 'V142', 'V157', 'V119', 'V120', 'V113', 'V123', 
 'V114', 'V116', 'V117', 'V199', 'id_24', 'id_25', 'id_07', 'id_08', 'id_21', 'id_26', 'id_27', 'id_23', 'id_22', 'id_28', 'D13', 'D7', 'D14', 'id_18', 'D12', 'D6', 'D7', 'D8', 'D12', 'D13', 'D14', 'C3']

def preprocess_data(df, freq_maps, agg_maps, l_enc, feature_columns):
   #df = df.drop(columns=ftnl, errors="ignore")
   df = apply_com(df)
   df = apply_freq(df, freq_maps)
   df = apply_AG(df, agg_maps)
   df = apply_le(df, l_enc)
   df = df.reindex(columns=feature_columns, fill_value=-1)

   return df