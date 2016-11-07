import pandas as pd
import numpy as np
import kh_config as config


class ErrorDetection():
    # def __init__(self, col_names_filename, data_filename, columns_description_filename):
    def __init__(self, data_filename, columns_description_filename):

        # self.df_col_names = pd.read_csv(col_names_filename, sep=',', encoding='utf-8', low_memory=False, header=None)
        self.df_cols_desc = pd.read_csv(columns_description_filename, sep=',', encoding='utf-8', low_memory=False, header=0)

        # get column to type map
        converters = {}
        self.colTypes = {}
        for row in self.df_cols_desc.iterrows():
            col = row[1]['Column']
            type_ = row[1]['Type']
            self.colTypes[col] = type_

            if type_ in ['PRIMARY_KEY', 'FOREIGN_KEY', 'ENUM', 'TEXT']:
                converters[col] = str

        # self.df_complete = pd.read_csv(data_filename, sep=',', encoding='utf-8', low_memory=False, header=None,
        #                                names=self.df_col_names.loc[0], converters=converters)
        # self.df_complete = pd.read_csv(data_filename, sep=',', low_memory=False, converters=converters)
        # self.df_complete = pd.read_csv(data_filename, sep=',', encoding='utf-8', low_memory=False, converters=converters)
        self.df_complete = pd.read_csv(data_filename, sep=',', encoding='latin-1', low_memory=False, converters=converters)


        self.unused_cols = list(self.df_cols_desc[self.df_cols_desc['Unused'] == True]['Column'])
        self.used_cols = list(self.df_cols_desc[self.df_cols_desc['Unused'] != True]['Column'])
        print('used cols: ', self.used_cols)

        self.materials = list(self.df_complete['Material'].unique())
        self.plants = list(self.df_complete['Plant'].unique())
        self.materialTypes = list(self.df_complete['Material Type'].unique())

        self.numeric_used_cols = []
        for col in self.used_cols:
            if self.colTypes[col] in ['INT', 'FLOAT']:
                self.numeric_used_cols += [col]

        # remove all cols that have only a single value
        self.colNumValues = []
        for col in self.used_cols:
            print('col is: ', col)
            if col not in 'Plant':
                values = self.df_complete[col].unique()
                if len(values) == 1:
                    self.unused_cols += [col]
                    self.used_cols.remove(col)
                else:
                    self.colNumValues += {(len(values), col)}
        self.colNumValues.sort()

        # remove unused columns
        self.df = self.df_complete[self.used_cols]

        # initialize suspicious values
        self.errors = []

    def writeCleanedData(self, filename):
        # save data as csv-file
        self.df.to_csv(filename, sep='\t', encoding='utf-8', index=False, columns=self.used_cols)

    def getValueCounter(self, values):

        valueCounter = {}
        total = len(values)
        for value in values:
            if pd.isnull(value):
                valueCounter[np.nan] = valueCounter.get(np.nan, 0) + 1
            else:
                valueCounter[value] = valueCounter.get(value, 0) + 1

        return (valueCounter, total)

    def getValueCounterList(self, values):

        valueCounter, total = self.getValueCounter(values)
        freqValues = []
        for k, v in valueCounter.items():
            freqValues += [(v, k)]
        freqValues.sort()

        return (freqValues, total)

# data_filename = config.data_file
# # test_filename = config.test
# columns_description_filename = config.column_description_file
# ed = ErrorDetection(data_filename, columns_description_filename)