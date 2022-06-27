import numpy as np
import pandas as pd
import scipy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from sklearn.utils.validation import check_is_fitted


class ExtractBoolean(BaseEstimator, TransformerMixin):
    
    r'''
    Extract the boolean columns of a pandas DataFrame and forward it along the sklearn pipeline.
    '''
    
    def __init__(self):
        self.columns = None
    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        self.columns = X.select_dtypes('bool').columns
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes('bool').replace({True: 1, False: 0})


class ExtractNumeric(BaseEstimator, TransformerMixin):
    
    r'''
    Extract the numeric columns of a pandas DataFrame and forward it along the sklearn pipeline.
    '''
    
    def __init__(self):
        self.columns = None
    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        self.columns = X.select_dtypes(np.number).columns
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(np.number)


class ExtractCategorical(BaseEstimator, TransformerMixin):
    
    r'''
    Extract the categorical columns of a pandas DataFrame and forward it along the sklearn pipeline.
    '''
    
    def __init__(self, returnDataFrame = False, sparse = False):
        self.onehotencoder = None
        self.columns = None
        self.returnDataFrame = returnDataFrame
        self.sparse = sparse
    def fit(self, X, y = None):
        assert isinstance(X, pd.DataFrame)
        categoricalDF = X.select_dtypes('category')
        self.columns = [column + '_' + str(cat) for column, cats in [(column, categoricalDF[column].cat.categories) for column in categoricalDF] for cat in cats]
        self.onehotencoder = OneHotEncoder(handle_unknown = 'ignore', categories = [categoricalDF[column].cat.categories for column in categoricalDF])
        self.onehotencoder.fit(categoricalDF)
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        processedDF = self.onehotencoder.transform(X.select_dtypes('category'))
        if not self.sparse:
            processedDF = processedDF.toarray()
        if self.returnDataFrame:
            if scipy.sparse.issparse(processedDF):
                processedDF = processedDF.toarray()
            processedDF = pd.DataFrame(processedDF)
            processedDF.columns = self.columns
        return processedDF


class GroupByImputer(BaseEstimator, TransformerMixin):

    r'''
    Impute missing data based on a "groupby" aggregation function.
    '''
    
    def __init__(self, groupby, target, agg = 'median', bins = None, copy = True):
        
        if isinstance(groupby, str):
            self.groupby = [groupby]
        elif isinstance(groupby, list):
            self.groupby = groupby
        else:
            raise TypeError('groupby must be a str or a list')

        if isinstance(target, str):
            self.target = [target]
        elif isinstance(target, list):
            self.target = target
        else:
            raise TypeError('target must be a string')
        
        if isinstance(agg, list):
            raise TypeError('agg cannot be a list')
        self.agg = agg
        
        if bins is None:
            self.bins = [None] * len(self.groupby)
        else:
            if isinstance(bins, list):
                if len(bins) == len(self.groupby):
                    self.bins = bins
                else:
                    raise Exception('bins and groupby must have same length')
            else:
                raise TypeError('bins is not a list')

        self.copy = copy

    def fit(self, X, y = None):
        
        assert isinstance(X, pd.DataFrame)
        groupByList = []
        for idx, column in enumerate(self.groupby):
            df = X[column]
            if (self.bins[idx] is not None) and (pd.api.types.is_numeric_dtype(df.dtype)):
                if isinstance(self.bins[idx], int):
                    # df = pd.qcut(df, self.bins[idx])
                    qbins = df.quantile(np.linspace(0, 1, self.bins[idx] + 1)).values
                    qbins[0] = float('-inf')
                    qbins[-1] = float('inf')
                    df = pd.cut(df, qbins)
                else:
                    df = pd.cut(df, self.bins[idx])
                self.bins[idx] = df.cat.categories
            groupByList.append(df)
        self.impute_table_ = X.groupby(groupByList).agg(self.agg)[self.target]
        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)
        check_is_fitted(self, 'impute_table_')
        if self.copy:
            X = X.copy()

        groupByList = []
        for bin, column in zip(self.bins, self.groupby):
            if bin is None:
                groupByList.append(column)
            else:
                groupByList.append(pd.cut(X[column], bin))
        grpby = X.groupby(groupByList)
        # X[self.target] = grpby[self.target].apply(lambda groupframe: groupframe.fillna({target: self.impute_table_.loc[groupframe.name, target] for target in self.target}))
        for target in self.target:
            X[target] = grpby[target].apply(lambda groupframe: groupframe.fillna(self.impute_table_.loc[groupframe.name, target]))
        return X


class CopyDataFrame(BaseEstimator, TransformerMixin):

    r'''
    Copy the pandas DataFrame.
    '''

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        return X


class SaveDataFrame(BaseEstimator, TransformerMixin):

    r'''
    Save the pandas DataFrame to a CSV file.
    '''

    def __init__(self, filename, format = 'csv'):
        self.filename = filename
        self.format = format

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        if self.format == 'csv':
            X.to_csv(self.filename)
        return X


class AddMissingIndicator(BaseEstimator, TransformerMixin):

    r'''
    Add a column to the DataFrame that indicates whether the data in another specified column is missing.
    '''

    def __init__(self, column, copy = True):
        self.column = column
        self.copy = copy
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.copy:
            X = X.copy()
        X['missing_' + self.column] = X[self.column].isnull()
        return X


class ApplyFunction(BaseEstimator, TransformerMixin):

    r'''
    Apply a function to the DataFrame.
    '''

    def __init__(self, func, copy = True, returnsDataFrame = True):
        self.func = func
        self.copy = copy
        self.returnsDataFrame = returnsDataFrame

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.copy:
            X = X.copy()
        if self.returnsDataFrame:
            X = self.func(X)
        else:
            self.func(X)
        return X


class DropColumn(BaseEstimator, TransformerMixin):

    r'''
    Delete a column in the DataFrame.
    '''
    
    def __init__(self, column):
        self.column = column

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.drop(columns = self.column, axis = 1)


class DataFrameEval(BaseEstimator, TransformerMixin):

    r'''
    Using pandas.DataFrame.eval, do data operations specified by evalString on the DataFrame.
    '''

    def __init__(self, evalString):
        self.evalString = evalString

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.eval(self.evalString)