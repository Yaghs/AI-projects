import sys
import numpy as np
import pandas as pd
import math

def compute_gaussian_probab(x, mean, var):
    res = 1
    for i in range(len(x)):
        exponent = math.exp(-((x[i]-mean[i])**2 / (2 * var[i] )))
        res *= (1 / (math.sqrt(2 * math.pi * var[i]))) * exponent
    return res

def main():
    #-------------wheat dataset-----------------#
    df=pd.read_csv("wheat.csv")   
    #---randomize data-----------------------#
    dfrandom = df.sample(frac=1, random_state=1119).reset_index(drop=True) 
   # data read from a file is read as a string, so convert the first 7 cols to float
    df1 = dfrandom.iloc[:,0:7].astype(float)
    #---separate out the last column
    df2 = dfrandom.iloc[:,7]
    #---combine the 4 numerical columns and the ast column that has the flower category 
    dfrandom = pd.concat([df1,df2],axis=1)
    print(dfrandom)
    #---separate the data into training and test parts by taking a large portion, atleast 80% percent 
    dftrain = dfrandom.iloc[0:170,:]
    dftest = dfrandom.iloc[170:,:]
    print(dftest)
     #---assemble the data by categories i.e., classes. Since there are 3 classes inside this dataset we will use all 3
    df_class1 = dfrandom[dfrandom['class'] == 1]
    print(df_class1)
    df_class2 = dfrandom[dfrandom['class'] == 2]
    print(df_class2)
    df_class3 = dfrandom[dfrandom['class'] == 3]
    print(df_class3)

    #---------find mean of each class---------
    mean_class1 = df_class1.iloc[:,0:7].mean(axis=0)
    print('mean class 1\n',mean_class1)
    mean_class2 = df_class2.iloc[:,0:7].mean(axis=0)
    print('mean class 2\n',mean_class2)
    mean_class3 = df_class3.iloc[:,0:7].mean(axis=0)
    print('mean class 3\n',mean_class3)


    #---------find variance of each class---------
    var_class1 = df_class1.iloc[:,0:7].var(axis=0)
    print('var class 1\n',var_class1)
    var_class2 = df_class2.iloc[:,0:7].var(axis=0)
    print('var class 2\n',var_class2)
    var_class3 = df_class3.iloc[:,0:7].var(axis=0)
    print('var class3\n',var_class3)

    #---do prediction on the test set via Naive Bayes
    count_correct = 0

    for i in range(len(dftest)):
        x = dftest.iloc[i,0:7].values

        probC1 = compute_gaussian_probab(x, mean_class1.values, var_class1.values)
        probC2 = compute_gaussian_probab(x, mean_class2.values, var_class2.values)
        probC3 = compute_gaussian_probab(x, mean_class3.values, var_class3.values)
        probs = np.array([probC1, probC2, probC3])
        maxindex = probs.argmax() + 1
       

        if dftest.iloc[i,7] == 'class 1':
            maxindex = 0
        elif dftest.iloc[i,7] == 'class 2':
            maxindex = 1
        elif dftest.iloc[i,7] == 'class 3':
            maxindex = 2

             
        if maxindex == dftest.iloc[i,7]:
            count_correct += 1
        
    print('Classification accuracy =', (count_correct / len(dftest)) * 100)
    


if __name__ == "__main__":
    sys.exit(int(main() or 0))