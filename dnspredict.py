import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import plotly.express as px
import pickle

from keras.layers import Dense, LSTM, Layer
from keras.regularizers import L1L2
from keras.models import load_model
from keras.models import Sequential, Model, Input
from keras import layers
from keras import activations,initializers,regularizers,constraints
from keras import backend as K

def prepData(filename,locs=[],filters={},index=[],return_locs=False):
    '''
    Prepare data into a aggregate count time series data frame from a csv file with following format for each line:,count,ip_version,location,protocol,record_name,record_type,time,domain

    :param filename: location of the input file in csv format
    :param locs: list of all locations to be included in the output, missing locations are included with 0 counts. If None, then the locations in the given file are used (default - [])
    :param filters: dictionary containing filters for each column, each filter rule is of the form: 'column_name': list of allowed values (default - {})
    :param index: external time index provided to reindex the data frame (default - [])
    :param return_locs: boolean indicator, if True, then the locs parameter is returned (default - False)
    :return: dataframe containing aggregated counts by location
    '''

    df_filtered = pd.read_csv(filename)
    for f,v in filters.items():
        df_filtered = df_filtered.loc[df_filtered[f].isin(v)]

    df_filtered['time_f'] = pd.to_datetime(df_filtered['time'])
    df_filtered.drop(columns=['time'],inplace=True)
    df_filtered.rename(columns={'time_f':'time'},inplace=True)
    df_agg = df_filtered.groupby(['time','location'])['count'].sum().unstack().fillna(0).sort_index()

    r = pd.date_range(df_agg.index.min(),df_agg.index.max(),freq='600s')
    df_agg = df_agg.reindex(r).fillna(0)
    
    if len(locs) == 0:
        locs = df_agg.columns
    else:
        locs_i = df_agg.columns
        locs_diff = list(set(locs).difference(set(locs_i)))
        df_rem = pd.DataFrame([],columns=locs_diff)
        df_agg = df_agg.join(df_rem,how='left')
    if len(index) != 0:
        df_agg = df_agg.reindex(index)
        df_agg.fillna(0,inplace=True)
    if return_locs:
        return df_agg,locs
    else:
        return df_agg
    
def prepDataJSON(filename,locs=[],filters={},index=[],return_locs=False):
    '''
    Prepare data into a aggregate count time series data frame from the json file containing query records.

    :param filename: location of the input file in json format
    :param locs: list of all locations to be included in the output, missing locations are included with 0 counts. If None, then the locations in the given file are used (default - [])
    :param filters: dictionary containing filters for each column, each filter rule is of the form: 'column_name': list of allowed values (default - {})
    :param index: external time index provided to reindex the data frame (default - [])
    :param return_locs: boolean indicator, if True, then the locs parameter is returned (default - False)
    :return: dataframe containing aggregated counts by location
    '''
    js_full = json.load(open(filename,'r'))
    domainName = js_full['requestInfo']['domainName']
    # first create meta dataframe
    meta = []
    for jl in js_full['data']:
        meta.append(dict([(c['queryParamName'],c['queryParamValue']) for c in jl['key']['components']]))
    df_meta = pd.DataFrame(meta)
    # filter out unwanted records
    for f,v in filters.items():
        if f in df_meta.columns:
            df_meta = df_meta.loc[df_meta[f].isin(v)]
    if len(locs) == 0:
        locs = sorted([int(x) for x in df_meta['location'].unique()])
    # create total counts dataframe
    df_locs = {}
    for _loc in locs:
        loc = str(_loc)
        loc_inds = df_meta.loc[df_meta['location'] == loc].index
        if len(loc_inds) > 0:
            dfs = []
            for loc_ind in loc_inds:
                df = pd.DataFrame(js_full['data'][loc_ind]['timeSeries']).rename(columns={'x':'time','y':loc})
                df['time'] = df['time'].map(lambda x: datetime.fromtimestamp(x/1000))
                dfs.append(df.set_index('time'))

            df_loc = dfs[0]
            for df in dfs[1:]:
                df_loc = df_loc.join(df,how='outer',lsuffix='_l')
            df_loc.fillna(0,inplace=True)
            df_loc = pd.DataFrame(df_loc.sum(axis=1),columns=[loc])
        else:
            df_loc = pd.DataFrame([],columns=[loc])
        df_locs[loc] = df_loc

    df_agg = df_locs[str(locs[0])]
    for i in range(1,len(locs)):
        df_agg = df_agg.join(df_locs[str(locs[i])],how='outer')
    df_agg.fillna(0,inplace=True)
    if len(index) != 0:
        df_agg = df_agg.reindex(index)
        df_agg.fillna(0,inplace=True)
    if return_locs:
        return df_agg,domainName,locs
    else:
        return df_agg,domainName

def dot_product(x, kernel):
    '''
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow

    :param x: input
    :param kernel: weights

    :return: None
    '''
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

def trim(arr,mult):
    return arr[0:mult*int(np.floor(arr.shape[0]/mult)),:]

def pad(arr,mult):
    return np.vstack([arr,np.zeros((mult*int(np.ceil(arr.shape[0]/mult)) - arr.shape[0],arr.shape[1]))])

def prepTFData(data, locs, include_nx=False, data_nx=None, history=1, num_timepoints=1, scaler=None):
    """
    Prepare the data in the tensorflow format.

    :input data: dataframe containing counts data for a given domain
    :input locs: list of target locations
    :input include_nx: flag to indicate if the nxdomain counts have to be included (default False)
    :input data_nx: optional dataframe containing nxdomain counts (required if ``include_nx = True``)
    :input history: length of history for the predictive model (default 1)
    :input num_timepoints: length of time series in each batch (default 1)
    :scaler: optional scaler tuple to scale the data (for creating test data)

    :return X: training input - A (n1 x n2 x n3) ``numpy`` array. n1 is equal to ``ceil(data.shape[0]/num_timepoints)``. X is padded if the number of rows in data is not an exact multiple of num_timepoints. n2 is equal to num_timepoints. n3 is equal to (history x length of locs) if ``include_nx==False``, otherwise (2 x history x length of locs).
    :return Y: training targets - A (n1 x n2 x n3) ``numpy`` array. n1 is equal to ``ceil(data.shape[0]/num_timepoints)``. Y is padded if the number of rows in data is not an exact multiple of num_timepoints. n2 is equal to num_timepoints. n3 is equal to (history x length of locs).
    :return scaler: minmax scaler tuple used to normalize data (only if scaler == None)
    """

    if include_nx and not type(data_nx) == pd.DataFrame:
        raise AssertionError('nx counts data frame required')
    data_sub = data.loc[:, locs]
    if include_nx:
        locs_nx = ['{}_nx'.format(l) for l in locs]
        data_nx_sub = data_nx.loc[:, locs_nx]
        data_sub = data_sub.join(data_nx_sub, how='left')
        data_sub.fillna(0, inplace=True)
    else:
        locs_nx = []
    
    col_names = list(data_sub.columns)
    cols = list()
    names = list()
    for i in range(history, 0, -1):
        cols.append(data_sub.shift(i))
        names += [str(name) + '(t-%d)' % i for name in col_names]

    X_df = pd.concat(cols, axis=1)
    X_df.columns = names
    X_df.dropna(inplace=True)
    cols = list()
    names = list()
    for name in col_names:
        if name not in locs_nx:
            names.append(str(name) + '(t)')
            cols.append(data_sub.iloc[history:, :].loc[:, name])

    Y_df = pd.concat(cols, axis=1)
    Y_df.columns = names
    Y_df.dropna(inplace=True)
    
    if type(scaler) != tuple:
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_Y = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler_X.fit_transform(X_df.values)
        scaled_Y = scaler_Y.fit_transform(Y_df.values)
        newScaler = True
    else:
        scaled_X = scaler[0].transform(X_df.values)
        scaled_Y = scaler[1].transform(Y_df.values)
        newScaler = False

    X_df = pd.DataFrame(scaled_X,index=X_df.index,columns=X_df.columns)
    Y_df = pd.DataFrame(scaled_Y,index=Y_df.index,columns=Y_df.columns)
    
    
    #first pad the values to make the X_df.shape[0] an exact multiple of num_timepoints
    X_arr = pad(X_df.values,num_timepoints)
    Y_arr = pad(Y_df.values,num_timepoints)
    num_batches = int(X_arr.shape[0]/num_timepoints)
    X_arr = X_arr.reshape((num_batches,num_timepoints,X_arr.shape[1]))
    Y_arr = Y_arr.reshape((num_batches,num_timepoints,Y_arr.shape[1]))
    if newScaler:
        return (X_arr, Y_arr, (scaler_X,scaler_Y))
    else:
        return (X_arr, Y_arr)


def buildLSTMModel(input_shape,output_units,n_units=8,n_layers=1,l1=0.01,l2=0.01,attention=False):
    ''' 
    Build the LSTM model for multivariate time series prediction
    
    :input input_shape: ``tuple`` (length of history,number of input features)
    :input output_units: number of targets in the multivariate output
    :input n_units: Dimensionality of the hidden vector for LSTM (default 8)
    :input n_layers: Number of LSTM layers in the model (default 1)
    :input l1: l1-regularization penalty (default 0.01)
    :input l2: l2-regularization penalty (default 0.01)
    :input attention: boolean flag to indicate inclusion of an attention layer (default False)

    :return: ``keras`` LSTM model
    '''
    
    batch_size = 1 # this is always fixed to 1
    stateful = True # this is always set to True
    model = Sequential()
    reg = L1L2(l1=l1, l2=l2)
    for i in range(n_layers):
        if i == 0:
            lstm_layer = LSTM(n_units, 
                              batch_input_shape=(batch_size, input_shape[0], input_shape[1]),
                              stateful=stateful,
                              return_sequences=True)
        elif i != n_layers-1:
            lstm_layer = LSTM(n_units, stateful=stateful,return_sequences=True)
        else:
            lstm_layer = LSTM(n_units, stateful=stateful,return_sequences=True)
        model.add(lstm_layer)
        if attention and i == 0:
            model.add(AttentionWithContext())
    model.add(Dense(output_units))
    model.compile(loss='mse', optimizer='adam')
    return model

def modelPredict(model,testX_arr,test_shape,test_locations,test_index,target_scaler):
    '''
    Generate predictions using a previously trained model on the given test data set
    
    :input model: ``keras`` model for predicting query counts
    :input testX_arr: Test input `numpy` array
    :input test_shape: Shape (tuple) of the original test data
    :input test_locations: List of target locations
    :input test_index: Time series index for test data
    :input target_scaler: Target scaler to rescale the predictions
    
    :return: ``pandas DataFrame`` containing predictions
    '''
    # test on the new data
    preds = model.predict(testX_arr,batch_size=1)
    # reshape the predictions
    preds = np.reshape(preds,test_shape)
    preds = target_scaler.inverse_transform(preds)
    df_preds = pd.DataFrame(preds,index=test_index,columns=test_locations)
    return df_preds

def estimateADParams(model,trainX_arr,trainY_arr,train_shape,target_scaler):
    '''
    Estimate the mean and standard deviation parameters obtained using the model on 
    the training data.
    
    :input model: ``keras`` model for predicting query counts
    :input trainX_arr: Input `numpy` array used for training
    :input trainY_arr: Target `numpy` array used for training
    :input train_shape: Shape (tuple) of the original training data
    :input target_scaler: Target scaler to rescale the predictions
    
    :return: tuple containing `pandas DataFrame` for location-wise residual means and standard deviation
    '''
    
    preds_train = model.predict(trainX_arr,batch_size=1)
    res = np.abs(preds_train - trainY_arr)
    # reshape the residues
    res = np.reshape(res,train_shape)
    res = target_scaler.inverse_transform(res)
    res_mean = res.mean(axis=0)
    res_stdev = res.std(axis=0)
    
    return (res_mean,res_stdev)

def detectAnomalies(df_test,df_preds,res_stdev,thresh=3):
    '''
    Assigns an anomaly label using the model predictions and the observed data.
    
    :input df_test: ``pandas DataFrame`` containing the observed data
    :input df_preds: ``pandas DataFrame`` containing the model predictions
    :input res_stdev: standard deviations for each location
    :input thresh: integer threshold to determine anomalies (default 3)
    
    :return: DataFrame containing anomaly labels
    '''
    
    res_test = np.abs(df_preds - df_test)
    ad = res_test/res_stdev
    return (ad > thresh).astype(int)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,return_attention=False, **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super().build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        #result = K.sum(weighted_input, axis=1)
        result = weighted_input
        if self.return_attention:
            return [result, a]
        return result 

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0],input_shape[1], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0],input_shape[1], input_shape[-1]
