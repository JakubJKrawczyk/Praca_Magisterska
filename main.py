from Custom.Data.DataHelper import DataHelper

# Load data
data = DataHelper.load_mat_file("Custom/Data/EEG_data/1.mat", True)

DataHelper.print_mat_content(data)

# Preprocess data


# build model