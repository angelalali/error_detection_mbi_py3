from os.path import join

# specify data folder
data_folder = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/data 4/'

# specify input files
# header_file = join(data_folder, 'data_header.csv')
# column_description_file = join(data_folder, 'Fields Required for MDM Algorithms.xlsx')
# column_description_file = join(data_folder, 'columns_description.csv')
column_description_file = join(data_folder, '11-4 columns description.csv')
# data_file = join(data_folder, '10-20 15k.csv')
data_file = join(data_folder, 'Champagne data.csv')

# data_file = join(data_folder, '10-20 3.3k.csv')
train = join(data_folder, 'train.csv')
test = join(data_folder, 'test.csv')

# specify output files
output_folder = '/Users/yisilala/Documents/IBM/projects/kraft heinz project/output/11-6 after champagne'
global_histogram_errors_file = join(output_folder, 'global_histogram_errors.csv')
decision_tree_errors_file = join(output_folder, 'decision_tree_errors.csv')
all_errors_file = join(output_folder, 'James_out.csv')
