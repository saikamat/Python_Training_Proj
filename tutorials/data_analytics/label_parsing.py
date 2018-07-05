'''
1. extract labels from one csv file
2. place that label column inside another csv file
3. append the labels of two csv files
4. do the same for the other four columns denoting ddos attacks
5. split the large file into 70-30 test and training.
'''

import pandas as pd
import glob
import os
def concatenateCSV(indir = r'C:\Users\Sai Kamat\Downloads\CSVs', outfile = r'D:\DDOS\Y.csv', outfileX = r'D:\DDOS\X.csv'):
    os.chdir(indir)
    filelist = glob.glob('*.csv')
    dfList = []
    dfXList = []
    cols = [' Label']
    colsX = [' Bwd Packet Length Std', ' Average Packet Size', ' Flow Duration', ' Flow IAT Std']
    for filename in filelist:
        print(filename)
        df = pd.read_csv(filename, usecols = cols, encoding='cp1252')
        dfList.append(df)
        dfX = pd.read_csv(filename, usecols = colsX, encoding='cp1252')
        dfXList.append(dfX)
    concatenated_data_frame = pd.concat(dfList, axis = 0)
    concatenated_data_frame.columns = cols
    concatenated_data_frame.to_csv(outfile, index = None)
    concatenated_data_frameX = pd.concat(dfXList, axis = 0)
    concatenated_data_frameX.columns = colsX
    concatenated_data_frameX.to_csv(outfileX, index = None)
concatenateCSV()
##3.3 concatenating mulptiple CSV's 
#def concatenateCSV(indir = r'D:\DDOS', outfile = r'D:\DDOS\result.csv'):
#    os.chdir(indir)
#    filelist = glob.glob('*.csv')
#    dfList = []
#    cols = ['labels']
#    for filename in filelist:
#        print(filename)
#        df = pd.read_csv(filename, usecols = cols)
#        dfList.append(df)
#    concatenated_data_frame = pd.concat(dfList, axis = 0)
#    concatenated_data_frame.columns = cols
#    concatenated_data_frame.to_csv(outfile, index = None)

#concatenateCSV()

#test_file_1 = r'C:\Users\Sai Kamat\Downloads\CSVs\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv';
#newCSV = r'D:\newCSV.csv'

#test_file_2 = r'C:\Users\Sai Kamat\Downloads\CSVs\Monday-WorkingHours.pcap_ISCX.csv'
#newCSV2 = r'D:\newCSV2.csv'

## 1. extract labels from one csv file
#df0 = pd.read_csv(test_file_1)
#print(df.head())
#print(df0[' Flow Duration'])

## 2. place that label column inside another csv file
#ndf = pd.read_csv(test_file_1, usecols = [' Label']).to_csv(newCSV, index = False)

#3.1 append the multiple csv files
#WORKED
#data_frames = [pd.read_csv(p) for p in (r'D:\DDOS\abc.csv',r'D:\DDOS\def.csv',r'D:\DDOS\pqr.csv')]
#merged_df = pd.concat(data_frames, axis=1)
#merged_df.to_csv(r'D:\DDOS\result.csv', index = False)

##3.2 append multiple csv files vertically
#WORKED
#data_frames = [pd.read_csv(p) for p in (r'D:\DDOS\abc.csv',r'D:\DDOS\def.csv',r'D:\DDOS\pqr.csv')]
#merged_df = pd.concat(data_frames, axis=0)
#merged_df.to_csv(r'D:\DDOS\result.csv', index = False)

##3.2 append multiple csv files vertically, but merge only specific columns
#data_frames = [pd.read_csv(p, usecols = ['labels']) for p in (r'D:\DDOS\abc.csv',r'D:\DDOS\def.csv',r'D:\DDOS\pqr.csv')]
#merged_df = pd.concat(data_frames, axis=0)
#merged_df.to_csv(r'D:\DDOS\result.csv', index = False)

##3.3 making the reading list
#merged_df = []
#for p in (r'D:\DDOS\abc.csv',r'D:\DDOS\def.csv',r'D:\DDOS\pqr.csv'):
#    data_frames = pd.read_csv(p, usecols = ['labels'])
#    merged_df = pd.concat(data_frames, axis=0)
#merged_df.to_csv(r'D:\DDOS\result.csv', index = False)

#df = pd.read_csv(r'C:\Users\Sai Kamat\Downloads\CSVs\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
#labels = df['Flow ID']
#filtered_csv = pd.read_csv(r'C:\Users\Sai Kamat\Downloads\CSVs\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols = ['Flow ID','Destination','Label'])
#df = pd.read_csv(r'C:\Users\Sai Kamat\source\repos\Python_for_DS\Datasets\loan_prediction\X_train.csv')
#df = pd.read_csv(r'C:\Users\Sai Kamat\source\repos\Python_for_DS\Datasets\loan_prediction\X_train.csv',sep='\s*,\s*',header=0, encoding='ascii', engine='python')
#labels = df['Loan ID']

##THIS WORKED.
#df = pd.read_csv(r'C:\Users\Sai Kamat\Desktop\imdbratings.csv')
#print(df.head())
#print(df.duration)

##THIS WORKED.
#df = pd.read_csv(r'C:\Users\Sai Kamat\Downloads\CSVs\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
## 1. extract labels from one csv file
#df0 = pd.read_csv(test_file_1)
#print(df.head())
#print(df0[' Flow Duration'])


##create new dataframe
#new_df = pd.DataFrame(columns=['Label'])

#data_from = pd.DataFrame.from_csv(test_file_1)
#data_to = pd.DataFrame.from_csv(r'D:\newCSV.csv', sep=',',parse_dates=False)

#all_files = glob.glob(file_path+'\*.csv')
#all_files = os.path.join(file_path, "\*.csv")
#frame = pd.DataFrame()
#list_ = []
#for file_ in all_files:
#    df = pd.read_csv(file_, usecols = [' Label'], header = 0)
#    list_.append(df)
#frame = pd.concat(list_)
#frame.to_csv(file_path+'\a.csv', sep='\t')
#frame.to_csv('D:\DDOS\abc.csv')