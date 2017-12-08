import pandas as pd
from sklearn import preprocessing

#import data as DataFrame
data = pd.read_csv("data_loan.csv",delimiter=";")
df_1=pd.DataFrame(data)

#create Series of attributes
str1=pd.Series(df_1.columns)

#drop attributes that have no one value
df=df_1.dropna(axis=1,how="all")

#compare previous attributes and remaining
str2=pd.Series(df.columns)
fin=str1[~str1.isin(str2)]

#save table with attributes list
fin.to_csv('list_del1.csv',encoding="utf-8",sep=";",index = False)
df.to_csv('data_del1.csv',encoding='utf-8',sep=';',index=False)

#drop post attributes
drop=pd.read_csv('drop.csv',names=['atr'])
drop=pd.DataFrame(drop)
for i in drop['atr']:
    del df[i]


#create the DFrame for description
description=pd.DataFrame(df.columns,columns=["переменные"])
list_type=[]
list_pass=[]
list_unique=[]
list_dpass=[]
list_duniq=[]
list_avg=[]
list_medians=[]
list_std=[]
list_min=[]
list_max=[]
m=len(df)

#fill description values
for i in description["переменные"]:
    list_type.append(df[i].dtype)
    list_pass.append(df[i].isnull().sum())
    if df[i].isnull().sum() >= 0.00001:
        list_dpass.append(df[i].isnull().sum() / m)
    else:
        list_dpass.append(0)
    list_unique.append(df[i].unique().size)
    if df[i].unique().size >= 0.000001:
        list_duniq.append(df[i].unique().size / m)
    else:
        list_duniq.append(0)
description["тип переменной"]=list_type
description["количество пропусков"]=list_pass
description["доля пропусков"]=list_dpass
description["уникальных элементов"]=list_unique
description["доля уникальных элементов"]=list_duniq
for i in description.index:
    if description.loc[i,"тип переменной"]!="O":
        list_avg.append(df[description.loc[i,"переменные"]].mean())
        list_medians.append(df[description.loc[i,"переменные"]].median())
        list_std.append(df[description.loc[i,"переменные"]].std())
        list_min.append(df[description.loc[i,"переменные"]].min())
        list_max.append(df[description.loc[i,"переменные"]].max())
    else:
        list_avg.append('-')
        list_medians.append("-")
        list_std.append("-")
        list_max.append("-")
        list_min.append("-")
    print(i, "признак")
description["тип переменной"]=list_type
description['тип переменной']=description['тип переменной'].replace(to_replace=['object','float64',"int64"],value=['атрибутивный','вещественный','целочисленный'])
description['среднее значение']=list_avg
description['медиана']=list_medians
description["стандартное отклонение"]=list_std
description["максимум"]=list_max
description["минимаум"]=list_min

#round values
description=description.round(4)

#save description table
description.to_csv("description.csv",sep=";",index = False)

#drop unnecessary attributs
del df["desc"]
del df['loan_status']
del df['sub_grade']

#create Series of attibutse labels
str1=pd.Series(df.columns)
m=len(df)

#create new description
description=pd.DataFrame(df.columns,columns=["переменные"])
list_type=[]
list_pass=[]
list_unique=[]
list_dpass=[]
list_duniq=[]
for i in description["переменные"]:
    list_type.append(df[i].dtype)
    list_pass.append(df[i].isnull().sum())
    if df[i].isnull().sum() >= 0.00001:
        list_dpass.append(df[i].isnull().sum() / m)
    else:
        list_dpass.append(0)
    list_unique.append(df[i].unique().size)
    if df[i].unique().size >= 0.000001:
        list_duniq.append(df[i].unique().size / m)
    else:
        list_duniq.append(0)
description["тип переменной"]=list_type
description["количество пропусков"]=list_pass
description["доля пропусков"]=list_dpass
description["уникальных элементов"]=list_unique
description["доля уникальных элементов"]=list_duniq

#del unsuitable attributes
for i in description.index:
    if description.loc[i,"доля пропусков"]>0.7:
        del df[description.loc[i,"переменные"]]
    if description.loc[i,"уникальных элементов"]>300 and \
            description.loc[i, "тип переменной"]=="O":
        del df[description.loc[i,"переменные"]]

#compare previous and new attributes lists
str2=pd.Series(df.columns)
fin=str1[~str1.isin(str2)]

#save list dropped attributes
fin.to_csv('fin_2_cleaning_short.csv',encoding="utf-8",sep=";",index = False)

#separate the target attribute
y=df["grade"].as_matrix()
del df["grade"]
df_y=pd.DataFrame(y,columns=["y"])

#transition to num
num,uniq=pd.factorize(df_y['y'])
df_y['y']=num
df_uniques_y=pd.DataFrame(uniq,columns=['previous'])
df_uniques_y.to_csv('uniques_y.csv',encoding="utf-8",sep=";")

#replace description dframe
description=pd.DataFrame(df.columns,columns=["переменные"])
list_type=[]
list_unique=[]
for i in description["переменные"]:
    list_type.append(df[i].dtype)
    list_unique.append(df[i].unique().size)
description["тип переменной"] = list_type
description["уникальных элементов"]=list_unique

#transition to boolean space for X
for i in description.index:
    if description.loc[i,"тип переменной"]=="O":
        uniq_list=df[description.loc[i,"переменные"]].unique()
        uniq_list=uniq_list.tolist()
        del uniq_list[-1]
        for j in uniq_list:
            d={j:1}
            print(str(description.loc[i,"переменные"]))
            print(j)
            df[str(description.loc[i,"переменные"])+"("+str(j)+")"]=df[description.loc[i,"переменные"]].map(d)\
                .replace(to_replace=['NaN'],value=0)
        del df[description.loc[i,"переменные"]]

#replce descrtption Dframe
description = pd.DataFrame(df.columns, columns=["переменные"])

#represent dframe as a matrix
X = df.as_matrix(columns=None)

#fill all NaNs with an mean for attribute
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)
df = pd.DataFrame(X, columns=description["переменные"])

#standardization data
columns=df.columns
df=preprocessing.StandardScaler().fit(df).transform(df)
df=pd.DataFrame(df,columns=columns)

#save the ready dataset
df.to_csv("X_loan_std.csv",index = False,encoding="utf-8",sep=";")
df_y.to_csv("y_loan.csv",index = False,encoding="utf-8",sep=";")