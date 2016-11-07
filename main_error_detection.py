import sys
from sklearn.tree.tree import DecisionTreeRegressor
# sys.path += ['./']

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import p_g_error_detection as pg

import sklearn.tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import kh_config as config

# import p_g_str_col_analysis as pg_str

# set global random seed to get reproducible results
np.random.seed(42)

# load error detection data handler
# col_names_filename = config.header_file
data_filename = config.data_file
columns_description_filename = config.column_description_file

# ed = pg.ErrorDetection(col_names_filename, data_filename, columns_description_filename)
ed = pg.ErrorDetection(data_filename, columns_description_filename)


###############################################################################
# STRING METRIC ANALYSIS
###############################################################################

# strAnalysis = pg_str.analyzeTable(ed.df, ['Maintenance status'])
# print(strAnalysis[:10])


###############################################################################
# GLOBAL HISTOGRAM ANALYSIS
###############################################################################

if len(ed.df) > 10000:
    # if there are less than < numCat >
    numDiffCat = 50
    # then a category is considered rare if it occurs less than or equal to <
    numRareCat = 10
else:
    # if there are less than < numCat >
    numDiffCat = 50
    # then a category is considered rare if it occurs less than or equal to <
    numRareCat = 5

print('Global Histogram Analysis...', end='')
sys.stdout.flush()

# provides the indices to be kept per column
colIndex = {}

# provides the values to be removed per column
colDroppedValues = {}

# initialize list of global histogram errors
globalHistogramErrors = []  # (material, plant, )

# analyze every column independently
for col in ed.used_cols:
    # print('col is: ', col)

    # skip special columns
    if col.strip() in ['Material', 'Plant Description', 'Description', 'Follow-up matl', 'Maintenance status']:
        continue

    # analyze numeric columns
    if ed.colTypes[col] in ['INT', 'FLOAT']:

        values = ed.df[col]
        if values.max() > 10: # if the numerical values are larger than 10, round them...
            values = round(values)
        values_histogram_map, _ = ed.getValueCounter(values)
        values_histogram, values_total = ed.getValueCounterList(values)

        # if the total number of different values is not too high

        if len(values.unique()) <= numDiffCat:
            dropped_values = []
            for vk in values_histogram:
                if vk[0] <= numRareCat:
                    dropped_values += [(0, vk[1])]
        else:
            non_zero_values = values[values != 0]
            q25 = np.percentile(non_zero_values, 25)
            q75 = np.percentile(non_zero_values, 75)
            iqr = q75 - q25
            ub = q75 + 2.2 * iqr

            dropped_values = []
            dropped_values_without_score = values[values > ub]
            for dropped_value in dropped_values_without_score:
                # if the value to be dropped occurs more than < numRareCat> times, don't drop it!
                if values_histogram_map[dropped_value] < numRareCat:
                    score = ub / dropped_value
                    dropped_values += [(score, dropped_value)]

        dropped_values.sort()
        colDroppedValues[col] = dropped_values
        colIndex[col] = ed.df[col].apply(lambda x: not x in [v[1] for v in dropped_values])

        continue

    # analyze date columns
    if ed.colTypes[col] in ['DATE']:
        # TODO
        colIndex[col] = ed.df[col].apply(lambda x: True)
        continue

    # analyze categorical columns
    valueCounterList, total = ed.getValueCounterList(ed.df[col])
    rareValues = []
    x = []
    y = []
    for ck in valueCounterList:
        count = ck[0]
        key = ck[1]
        x += [key]
        y += [count]
        if count <= numRareCat:
            rareValues += [key]

    colIndex[col] = ed.df[col].apply(lambda x: not x in rareValues)
    dropped_values = []
    colDroppedValues[col] = dropped_values
    if len(rareValues) > 0:
        x_ = np.array(range(len(y)))
        y = np.array(y)
        for rareValue in rareValues:
            dropped_values += [(0, rareValue)]

        plot = False
        if plot:
            plt.bar(x_ - 0.4, y / total)
            plt.plot(x_, np.cumsum(y / total), 'r', linewidth=3)
            plt.grid()
            plt.xticks(x_, x[:len(y)])
            plt.title(col)
            plt.show()

    for row in ed.df[colIndex[col] == False].iterrows():
        score = 0
        material = row[1]['Material']
        plant = row[1]['Plant']
        comment = 'Global Histogram Analysis'
        value = row[1][col]
        recommendation = ''
        globalHistogramErrors += [(score, material, plant, col, comment, value, recommendation)]

new_columns = ['cell', 'cell name', 'cluster', 'cognitive score 1', 'cognitive score 2', 'comment', 'advice']
all_columns = np.append(ed.df_complete.columns, new_columns)
df_global_histogram_errors = pd.DataFrame(columns=all_columns)

# transform error list to material -> plant -> list
materialPlantErrorList = {}
for error in globalHistogramErrors:
    # (score, material, plant, col, comment, value, recommendation)
    material = error[1]
    plant = error[2]
    errorList = materialPlantErrorList.setdefault(material, {}).setdefault(plant, [])
    errorList += [error]
    # print('error is: ', error)

for row in ed.df_complete.iterrows():
    print('row is: ', row[0])

    material = row[1]['Material']
    plant = row[1]['Plant']
    errorList = materialPlantErrorList.get(material, {}).get(plant, [])

    for error in errorList:

        newRow = row[1]
        newRow['cell name'] = error[3]
        newRow['cognitive score 2'] = int(round(9.0 * (1.0 - error[0]) + 1.0))
        newRow['comment'] = error[4]
        newRow['advice'] = error[6]

        df_global_histogram_errors = df_global_histogram_errors.append(newRow)

    # print('row is: ', row)


df_global_histogram_errors['cognitive score 2'] = df_global_histogram_errors['cognitive score 2'].astype(int)
df_global_histogram_errors = df_global_histogram_errors.fillna('')
# df_global_histogram_errors.to_csv(config.global_histogram_errors_file, index=False, columns=all_columns)

print(' Done.')

###############################################################################
# DECISION TREE ANALYSIS
###############################################################################

if len(ed.df) > 10000:
    minSamplesPerLeave = 100
    maxLeafNodes = 25
    requiredAccuracy = 0.9 # only consider decision trees with a minimal required accuracy
    minScore = 0.9 # only consider leaves that have a minimal confidence
    numSigmas = 6 # number of +/- sigmas to consider values as outliers
else:
    minSamplesPerLeave = 20
    maxLeafNodes = 10
    requiredAccuracy = 0.75 # only consider decision trees with a minimal required accuracy
    minScore = 0.8999 # only consider leaves that have a minimal confidence
    numSigmas = 3 # number of +/- sigmas to consider values as outliers

print('Decision Tree Analysis...', end='')
sys.stdout.flush()

ed.df = ed.df.fillna(value=0)

maintenanceStatusFeatures = []
colEncoder = {}
def relabeledDataFrame(df, colTypes):
    df_relabeled = pd.DataFrame()
    
    df_relabeled['Material'] = df['Material']
    df_relabeled['Plant'] = df['Plant']
    
    for col in df.columns:
        
        if col.strip() in ['Material', 'Plant Description', 'Description', 'Follow-up matl', 'Maintenance status']:
            continue
        #
        # if col == 'Maintenance status':
        #
        #     # extract features
        #     features = set()
        #     for s in df[col]:
        #         for c in s:
        #             features.add(c)
        #     features = list(features)
        #     features.sort()
        #     maintenanceStatusFeatures = features
        #
        #     # for every feature add a column
        #     for feature in features:
        #         df_relabeled[col + '_' + feature] = df[col].apply(lambda s: int(feature in s))
        #     continue
        #
        if colTypes[col] in ['INT', 'FLOAT']:
            df_relabeled[col] = df[col]
        if colTypes[col] == 'BOOLEAN':
            df_relabeled[col] = df[col].apply(lambda x: int(x == 'X' or x == 1))
        if colTypes[col] in ['PRIMARY_KEY', 'FOREIGN_KEY']:
            values = df[col].unique()
            if len(values) <= 25:
                for value in values:
                    df_relabeled[col + '_' + value] = df[col].apply(lambda x: int(x == value))
            else:
                encoder = preprocessing.LabelEncoder().fit(values)
                colEncoder[col] = encoder
                df_relabeled[col] = encoder.transform(df[col])
                pass
    
    return df_relabeled
   
X = relabeledDataFrame(ed.df, ed.colTypes)


df_dec_tree_errors = pd.DataFrame(columns=all_columns)
for col in ed.df.columns:
    print('column now:', col)
    
    # skip some columns
    if col.strip() in ['Material', 'Plant', 'Description', 'Follow-up matl', 'Material Type', 'MRP Controller',
                       'Maintenance status', 'Plant Description', 'Lot size', 'MRP group']:
        continue

    ############################################
#
#     if col == 'Maintenance status':
#
#         # get features
#         features = maintenanceStatusFeatures
#
#         # learn features independently
#         for feature in features:
#
# #             clf = RandomForestClassifier(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)
# #             Y_orig = ed.df[index][col + '_' + feature]
# #             Y = clf.predict(X[index][cols])
# #             correct = np.array(Y_orig == Y)
#
#             pass
#
#         continue
    
    ############################################
    
    # get column index and columns
    index = colIndex[col]
    cols = [c for c in X.columns if not col in c.split('_')[0]]
    cols.remove('Material')
    cols.remove('Plant')

    # only fit columns with more than one value
    values = ed.df[index][col].unique()
    if len(values) == 1:
        continue
        
    # analyze numeric columns
    if ed.colTypes[col] in ['INT', 'FLOAT']:
        clf = RandomForestRegressor(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)
        # clf = DecisionTreeRegressor(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)
        
        Y_orig = ed.df[index][col]
        clf = clf.fit(X[index][cols], Y_orig)
        Y = clf.predict(X[index][cols])
        
        # sklearn.tree.export_graphviz(clf, out_file='./dot/' + col + '.dot', feature_names=cols)

        errors = np.array(Y - Y_orig)
        mu = np.mean(errors)
        sigma = np.sqrt(np.var(errors))
        errors_scaled = (errors - mu) / sigma

        # 6 sigma bounds on scaled errors
        lb = -numSigmas
        ub = +numSigmas
        
        error_index = (errors_scaled < lb) | (errors_scaled > ub)
        
        if sum(error_index) == 0:
            continue
        
        plot = False
        if plot:
            plt.hist(errors_scaled)
            ylim = plt.ylim()
            plt.plot([lb, lb], ylim)
            plt.plot([ub, ub], ylim)
            plt.ylim(ylim)
            plt.grid()
            plt.title(col)
            plt.show()
        
            print('num errors:', sum(error_index))
        
        df_index = ed.df_complete[index]
        df_index_not_correct = df_index[error_index].copy(deep=True).reset_index()
        Y_not_correct = Y[error_index].copy()
        score = errors_scaled[error_index]
        
        df_index_not_correct['cell name'] = col
        df_index_not_correct['comment'] = 'Decision Tree Analysis (numerical)'
        df_index_not_correct['advice'] = Y_not_correct
        if col == 'FLOAT':
            df_index_not_correct['advice'] = df_index_not_correct['advice'].apply(lambda x: "%.3f" % x)
        else:
            df_index_not_correct['advice'] = df_index_not_correct['advice'].apply(lambda x: int(round(x)))
        divisor = max(score) - min(score) + 1
        df_index_not_correct['cognitive score 2'] = np.round(9.0 * (score - min(score) + 1) / divisor + 1)
        
        df_dec_tree_errors = df_dec_tree_errors.append(df_index_not_correct)
        
        continue
    
    # analyze date columns
    if ed.colTypes[col] in ['DATE']:
        # TODO
        continue
    
    # analyze categorical columns
    clf = RandomForestClassifier(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)
    
    Y_orig = ed.df[index][col]
    if ed.colTypes[col] == 'BOOLEAN':
        Y_orig = Y_orig.apply(lambda x: x == 'X' or x == '1')
    
    clf = clf.fit(X[index][cols], Y_orig)
    
    Y = clf.predict(X[index][cols])
    P = clf.predict_proba(X[index][cols])
    correct = np.array(Y_orig == Y)
    hit_rate = sum(correct) / len(correct)
    
    # only consider columns that can be predicted with a certain quality
    print('hit rate is: ')
    print(col, hit_rate)
    if hit_rate < requiredAccuracy:
        continue
    elif hit_rate == 1:
        continue
    
    df_index = ed.df_complete[index]
    df_index_not_correct = df_index[correct == False].copy(deep=True).reset_index()
    Y_not_correct = Y[correct == False].copy()
    P_not_correct = P[correct == False].copy()
    
    classIds = {}
    for i in range(len(clf.classes_)):
        classIds[clf.classes_[i]] = i
    setIds = np.vectorize(lambda c: classIds[c])
    
    Y_not_correct_ids = setIds(Y_not_correct)
    
    df_index_not_correct['cell name'] = col
    df_index_not_correct['comment'] = 'Decision Tree Analysis (categorical)'
    df_index_not_correct['advice'] = Y_not_correct
    
    score = []
    for i in range(len(P_not_correct)):
        score += [P_not_correct[i][Y_not_correct_ids[i]]]
    score = np.array(score)
    divisor = max(score) - min(score) + 1
    df_index_not_correct['cognitive score 2'] = (np.round(9.0 * (score - min(score) + 1) / divisor + 1))
    
    # filter on those where the cognitive score is above a certain threshold
    df_index_not_correct = df_index_not_correct[score >= minScore]
    df_dec_tree_errors = df_dec_tree_errors.append(df_index_not_correct)
    
try:
    df_dec_tree_errors['cognitive score 2'] = df_dec_tree_errors['cognitive score 2'].astype(int)
except:
    print('Exception: cannot convert cognitive score to int!')
    
df_dec_tree_errors = df_dec_tree_errors.fillna('')
# df_dec_tree_errors.to_csv(config.decision_tree_errors_file, index=False, columns=all_columns)

print('    Done.')

###############################################################################
# MERGE RESULTS
###############################################################################

print('Merge Results...', end='')
sys.stdout.flush()

df_all_errors = pd.DataFrame(columns = df_global_histogram_errors.columns)
df_all_errors = df_all_errors.append(df_global_histogram_errors)
df_all_errors = df_all_errors.append(df_dec_tree_errors)

try:
    df_all_errors['cognitive score 2'] = df_all_errors['cognitive score 2'].astype(int)
except:
    print('Exception: cannot convert cognitive score to int!')


df_all_errors.to_csv(config.all_errors_file, index=False, header=None, columns=all_columns)

print('             Done.')

