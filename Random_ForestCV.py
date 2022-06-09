from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from random import shuffle, randint
from time import time
import numpy as np
import os
import joblib as jl

# Pour save les classifieurs j'utilise joblib maintenant vu qu'on peut compresser les fichiers sinon j'avais des RF de 6Go

cwd = os.getcwd()
startTotal = time()
rand = randint(0,10000)*randint(0,10000)

"""
test = SelectKBest(score_func=chi2, k=100)
fit = test.fit(rows, listOrgans)
img.append(fit.transform(rows))
"""

def createDataSets(files):
    f = open(cwd+'\\fRanking.txt' if '\\' in cwd else cwd+'/fRanking.txt')
    csvreader = csv.reader(f)
    tab = []
    for i in csvreader:
        tab = i
    labels = []
    features = []
    for i in files:
        print("Reading data from file "+i)
        theFile = open(cwd+'\\Patients\\'+i if '\\' in cwd else cwd+'/Patients/'+i)
        csvreader = csv.reader(theFile)
        for row in csvreader:
            img = []
            for j in range(len(tab)):
                if tab[j] == 'True' : img.append(float(row[j]))
            features.append(img)
            labels.append(organs[int(row[0])])
    print("Done reading data")
    return features, labels
   

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

lo = sorted([organs[i] for i in organs])
bestRF = []

print("Starting...\n")

obj = os.scandir(cwd+'\\Patients\\' if '\\' in cwd else cwd+'/Patients/')
list_files = [i.name for i in obj]
print("4 Folds to be done : ")
testingPrecisions = []
trainingPrecisions = []

for i in range(4):
    startFold = time()
    print("\n---------------------------------------------------------------------------------------------------------\n")

    # GETTING FILES FOR TRAINING SET AND TEST SET

    print(f"Starting Fold number {i+1} : ")
    forest = RandomForestClassifier(n_estimators=300, n_jobs=2)
    testFiles = list_files[i*5:i*5+5]
    trainFiles = [trainFile for trainFile in list_files if trainFile not in testFiles]

    # TRAINING CLASSIFIER 

    print("\nStarting Training for this Fold")
    X_train, y_train = createDataSets(trainFiles)
    print("\nCreating RF\n")
    del trainFiles
    forest.fit(X_train,y_train)
    train_accuracy = metrics.precision_score(y_train,forest.predict(X_train),average='macro')
    print("Train Data Precision : {} %".format(round((train_accuracy*100),2)))
    trainingPrecisions.append(round((train_accuracy*100),2))
    del X_train
    del y_train 

    # TESTING CLASSIFIER 

    print("\nStarting Testing for this Fold")
    X_test, y_test = createDataSets(testFiles)
    del testFiles 
    predicted = forest.predict(X_test)
    testPrecision = round((metrics.precision_score(y_test,predicted,average='macro'))*100, 2)
    testingPrecisions.append(testPrecision)

    # PRINTING ACCURACY FOR ALL LABELS

    print(
        f"\nClassification report for classifier {forest} for fold number {i+1} :\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    
    # PLOTTING THE CONFUSION MATRIX OF EACH FOLD

    confusionMX = metrics.confusion_matrix(y_test,predicted)
    endFold = int(time() - startFold)
    print(f"Fold done in {endFold//60}m{endFold%60}s")

    confusionMX = metrics.confusion_matrix(y_test,predicted)
    plt.figure(figsize=(10,10))
    plt.imshow(confusionMX,cmap='rainbow_r')
    plt.title("Confusion Matrix for test Data of fold number "+str(i+1), fontsize=20)
    plt.xticks(np.arange(23),lo, rotation=90)
    plt.yticks(np.arange(23),lo)
    plt.ylabel('Actual Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.colorbar()
    width,height = confusionMX.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(confusionMX[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
    plt.savefig(cwd+"\\Results\\Matrix\\"+f"MatrixCV{rand}_Fold"+str(i+1)+".png" 
                if "\\" in cwd else cwd+"/Results/Matrix/"+f"MatrixCV{rand}_Fold"+str(i+1)+".png", bbox_inches='tight', dpi=300)
        
    # KEEP BEST RF OUT OF THE 4       (Plus d'actualité vu qu'on garde pas le meilleur)

    """if i == 0:
        bestRF = [forest, testPrecision]
    elif testPrecision > bestRF[1]:
        bestRF = [forest, testPrecision]
        print("\nNEW BEST FOREST")"""
    
    # SAVING FORESTS FOR NEXT STEP      (Je save les 4 et dans le code finalRF3D.py je donne le droit de choisir quel fichier prendre)

    print("Saving the best RF in the folder \"cvClassifiers\"")
    filename = cwd+f"\\cvClassifiers\\rfFold{i+1}.joblib" if "\\" in cwd else cwd+f"/cvClassifiers/rfFold{i+1}.joblib"
    jl.dump(forest, filename, compress=3)

filename = cwd+f"\\Results\\precisionsCV{rand}.txt" if "\\" in cwd else cwd+f"/Results/precisionsCV{rand}.txt"

with open(filename, 'w') as f:
    f.write("Training Precision, Testing Accuracy")
    print("\nTraining Precision, Testing Accuracy")
    for i,j in zip(trainingPrecisions, testingPrecisions):
        print("           ",i,", ",j)
        f.write(f"           {i} , {j}")
    print("           ",sum(trainingPrecisions)/4,", ",sum(testingPrecisions)/4)
    f.write(f"            {sum(trainingPrecisions)/4}, {sum(testingPrecisions)/4}")
end = int(time() - startTotal)
print(f"\nDone in {end//60}m{end%60}s")
