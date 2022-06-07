import csv
from random import shuffle, choice
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os

cwd = os.getcwd()

organs = {
            1247 : "Trachea",
            1302 : "Right Lung",
            1326 : "Left Lung",
            170 : "Pancreas",
            187 : "Gallbladder",
            237 : "Urinary Bladder",
            2473 : "Sternum",
            29193 : "First Lumbar Vertebra",
            29662 : "Right Kidney",
            29663 : "Left Kidney",
            30324 : "Right Adrenal Gland",
            30325 : "Left Adrenal Gland",
            32248 : "Right Psoas Major",
            32249 : "Left Psoas Major",
            40357 : "Muscle body of right rectus abdominis",
            40358 : "Muscle body of left rectus abdominis",
            480 : "Aorta",
            58 : "Liver",
            7578 : "Thyroid Gland",
            86 : "Spleen",
            0 : "Background",
            1 : "Body Envelope",
            2 : "Thorax-Abdomen"
        }

list_organs = []
rows = []
obj = os.scandir(cwd+'\\Patients\\')
list_files = [i.name for i in obj]
shuffle(list_files)
del list_files[0:17]
for i in list_files:
    f = open(cwd+'\\Patients\\'+i if '\\' in cwd else cwd+'/Patients/'+i)
    csvreader = csv.reader(f)
    for row in csvreader:
        rows.append(list(map(float,row[1:])))
        list_organs.append(organs[float(row[0])])

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=200, step=50)
fit = rfe.fit(rows, list_organs)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
f = open("fRanking.txt","w+")
for i in fit.support_:
    f.write(str(i)+",")

