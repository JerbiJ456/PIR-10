import nibabel as nib
import os
from random import choice
import numpy as np
from time import time
import os
import pickle
import csv
from math import dist

def translate(value):
    leftMax = 2976
    leftMin = -1024
    rightMin = 0
    rightMax = 255
    
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return round(rightMin + (valueScaled * rightSpan), 5)

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
            40357 : "Right rectus abdominis",
            40358 : "Left rectus abdominis",
            480 : "Aorta",
            58 : "Liver",
            7578 : "Thyroid Gland",
            86 : "Spleen",
            0 : "Background",
            1 : "Body Envelope",
            2 : "Thorax-Abdomen"
        }

cwd = os.getcwd()
start = time()

f = open(cwd+'\\fRanking.txt' if '\\' in cwd else cwd+'/fRanking.txt')
csvreader = csv.reader(f)
tab = []
for i in csvreader:
    tab = i

print("LOADING RF\n")
obj = os.scandir(cwd+'\\cvClassifiers\\' if '\\' in cwd else cwd+'/cvClassifiers/')
listRF = [i.name for i in obj]
for i in range(len(listRF)):
    print(f"{i}. {listRF[i]}")
foldChoisi = ""
while foldChoisi not in ['0','1','2','3']:
    foldChoisi = input("Which Fold do want to use : ")
filename = cwd+f"\\cvClassifiers\\{listRF[int(foldChoisi)]}" if "\\" in cwd else cwd+f"/cvClassifiers/{listRF[int(foldChoisi)]}"
with open(filename, 'rb') as file:  
    classifier = pickle.load(file)
print("\nFINISHED LOADING")

obj = os.scandir(cwd+'\\Patients\\' if '\\' in cwd else cwd+'/Patients/')
listTestFiles = [i.name.split('.')[0] for i in obj]
listTestFiles = listTestFiles[int(foldChoisi)*5:int(foldChoisi)*5+5]
print(listTestFiles)

lo = sorted([organs[i] for i in organs])

coor = {}
for i in lo:
    coor[i] = [0,0,0,0]

prob = {}
for i in lo:
    prob[i] = 0.0 

obj = os.scandir(cwd+'\\Patients3D\\' if '\\' in cwd else cwd+'/Patients3D/')
list_files = [i.name for i in obj if i.name.split('.')[0] in listTestFiles]
print(list_files)
fread = choice(list_files)
obj = os.scandir(cwd+'\\centers\\' if '\\' in cwd else cwd+'/centers/')
list_center_files = [i.name for i in obj if fread.split('.')[0] in i.name]
print(f"File Read : {fread}")

epi_img = nib.load(cwd+'\\Patients3D\\'+fread if '\\' in cwd else cwd+'/Patients3D/'+fread)
epi_img_data = epi_img.get_fdata()
tot_img = 0

for x in range(0,epi_img_data.shape[2]):
    for i in range(0,epi_img_data.shape[0]//33):
        for j in range(0,epi_img_data.shape[1]//33):
            imagette = np.array([])
            imagette = np.append(imagette, epi_img_data[i*33:i*33+33, j*33:j*33+33, x])
            imagette = [a for a,b in zip(imagette,tab) if b == 'True']
            for z in range(200):
                imagette[z] = translate(imagette[z])
            imagette = np.array([imagette])
            predicted = classifier.predict(imagette)
            prob_predicted = classifier.predict_proba(imagette)
            tot_img += 1
            for a,b in zip(lo,range(23)):
                prob[a] += prob_predicted[0][b]
            for a in predicted:
                coor[a][0] += i*33+(33/2)
                coor[a][1] += j*33+(33/2)
                coor[a][2] += x
                coor[a][3] += 1
    print(f"z = {x} done, {epi_img_data.shape[2]-x-1} left", end='\r')

print("\n\nCenter Coordinates : \n".upper())
for i in coor:
    key = [k for k, v in organs.items() if v == i][0]
    for j in list_center_files: 
        if '_'+str(key)+'_' in j:
            f = open(cwd+'\\centers\\'+j if '\\' in cwd else cwd+'/centers/'+j)
            csvreader = csv.reader(f)
            for z in csvreader:
                organCoor = list(map(float,z))
    if coor[i][3]:
        x = round(coor[i][0]/coor[i][3],2)
        y = round(coor[i][1]/coor[i][3],2)
        z = round(coor[i][2]/coor[i][3],2)
        print(f"{i} : ({x},{y},{z}), Real Center : {(organCoor[0],organCoor[1],organCoor[2])} d = {round(dist((x,y,z),(organCoor[0],organCoor[1],organCoor[2])),2)}")
    else:
        print(f"Nothing for {i}")

tot = 0
print("\nProbabilities : \n".upper())
for i in prob:
    print(f"Pr({i}|imagette) = {round(prob[i]*100/tot_img,2)}")
    tot += prob[i]/tot_img
print(f"Ptot(organe|imagette) = {round(tot*100,2)}")