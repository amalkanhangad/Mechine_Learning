#one hot encoding
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

poke_df = pd.read_csv('pokemon.csv',encoding='utf-8')

gen_le = LabelEncoder()
gen_labels = gen_le.fit_transform(poke_df['Generation'])
poke_df['Gen_Label'] = gen_labels

leg_le =LabelEncoder()
leg_labels = leg_le.fit_transform(poke_df['Legendary'])
poke_df['Lgnd_label']= leg_labels

poke_df_sub = poke_df[['Name','Generation','Legendary','Gen_Label','Lgnd_label']]
poke_df_sub.iloc[4:10]

gen_onehot_features = pd.get_dummies(poke_df['Generation'])
print(pd.concat([poke_df[['Name','Generation']],gen_onehot_features],axis=1).iloc[4:10])

leg_ohe = OneHotEncoder()
leg_feature_arr = leg_ohe.fit_transform(poke_df[['Lgnd_label']]).toarray()
leg_feature_labels = ['Legendary_'+str(cls_label) for cls_label in leg_le.classes_]
leg_features = pd.DataFrame(leg_feature_arr,columns=leg_feature_labels)

gen_ohe = OneHotEncoder()
gen_feature_arr = leg_ohe.fit_transform(poke_df[['Gen_Label']]).toarray()
gen_feature_labels = list(gen_le.classes_)
gen_features = pd.DataFrame(gen_feature_arr,columns=gen_feature_labels)
poke_df_ohe = pd.concat([poke_df_sub,gen_features,leg_features],axis=1)
columns = sum([['Name','Generation','Gen_Label'],gen_feature_labels,['Legendary','Lgnd_label'],leg_feature_labels],[])
print(poke_df_ohe[columns].iloc[4:10])