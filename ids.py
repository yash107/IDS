import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model
from keras.models import Sequential  #for building the lstm model
from keras.layers import LSTM, Dense  # for adding the lstm layer
import matplotlib.pyplot as plt    # for plotting 

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

# DATA PREPROCESSING 
dataset=pd.read_csv(r"C:\Users\admin\Desktop\mini\KDDTrain+.txt", names=col_names)  #importing the dataset
dataset.drop(['difficulty_level'], axis=1, inplace=True)  #dropping a column named difficulty_level
dataset.shape
dataset['label'].value_counts()  # now there are only 4 labels in the dataset  1.'Dos', 2.'R2L', 3.'Probe', and 4.'U2R'

# this functions basically replace all the labels in mainly 4 categories like DOS,R2L,PROVE,U2R
def change_label(df):
  df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
  df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
       'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
  df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
  df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)

# making the data suitable for the model  
change_label(dataset)
dataset.label.value_counts()
numeric_col = dataset.select_dtypes(include='number').columns
print(numeric_col)
print(numeric_col.shape)
# rest_col=dataset.columns.difference(numeric_col)
# print(rest_col)
# print(rest_col.shape)

std_scaler = preprocessing.StandardScaler()
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
  return df
dataset = normalization(dataset.copy(),numeric_col)
dataset

# selecting categorical data attributes
cat_col = ['protocol_type','service','flag']
     

# creating a dataframe with only categorical attributes
categorical = dataset[cat_col]
categorical.head()

# one-hot-encoding categorical attributes using pandas.get_dummies() function
categorical = pd.get_dummies(categorical,columns=cat_col)
categorical.head()
bin_label = pd.DataFrame(dataset.label.map(lambda x:'normal' if x=='normal' else 'abnormal'))
bin_data = dataset.copy()
bin_data['label'] = bin_label

le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['intrusion'] = enc_label
bin_data.head()
bin_data = pd.get_dummies(bin_data,columns=['label'],prefix="",prefix_sep="") 
bin_data['label'] = bin_label
bin_data
plt.figure(figsize=(8,8))
plt.pie(bin_data.label.value_counts(),labels=bin_data.label.unique(),autopct='%0.2f%%')
plt.title("Pie chart distribution of normal and abnormal labels")
plt.legend()
plt.show() 

# creating a dataframe with only numeric attributes of binary class dataset and encoded label attribute 
numeric_bin = bin_data[numeric_col]
numeric_bin['intrusion'] = bin_data['intrusion']
corr= numeric_bin.corr() # dependency using correlqation matrix
corr_y = abs(corr['intrusion'])
highest_corr = corr_y[corr_y >0.5] #more dependent column with intusion column
highest_corr.sort_values(ascending=True)
numeric_bin = bin_data[['count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                         'logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]

# joining the selected attribute with the one-hot-encoded categorical dataframe
numeric_bin = numeric_bin.join(categorical)
# then joining encoded, one-hot-encoded, and original attack label attribute
bin_data = numeric_bin.join(bin_data[['intrusion','abnormal','normal','label']])
# saving final dataset to disk
# final dataset for binary classification
bin_data
X = bin_data.iloc[:,0:93] 
Y = bin_data[['intrusion']] 
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)
X_train = X_train.values  
y_train = np.array(y_train)
x_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
# Assuming x_train and y_train are defined earlier in your code
x_train = np.array(x_train, dtype=np.float32)

# Corrected line using built-in int
y_train = np.array(y_train, dtype=int)
print(x_train.shape)
x_train.shape
# 1 layer lstm model  LONG SHORT TERM MEMORY
lst = Sequential()
# input layer and LSTM layer with 50 neurons
lst.add(LSTM(50,input_dim=93))
# outpute layer with sigmoid activation as it gives the probablity in the range of 0-1 thus suitaible for our model
lst.add(Dense(1,activation='sigmoid'))
lst.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
lst.summary()

history = lst.fit(x_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)
X_test = np.array(X_test)
x_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=int)
test_results = lst.evaluate(x_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
#plt.savefig('plots/lstm_binary_accuracy.png')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Plot of loss vs epoch for train and test dataset")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
# plt.savefig('plots/lstm_binary_loss.png')
plt.show()


