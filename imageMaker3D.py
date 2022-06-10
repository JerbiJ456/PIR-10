import nibabel as nib
import os
from random import shuffle, randint, choice
from PIL import Image, ImageDraw
import numpy as np
from time import time
from matplotlib import colors
import csv

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
            480 : "Aorta",
            1247 : "Trachea",
            1302 : "Right Lung",
            1326 : "Left Lung",
            170 : "Pancreas",
            187 : "Gallbladder",
            237 : "Urinary Bladder",
            2473 : "Sternum",
            29193 : "First Lumbar Vertebra",
            29662 : "Right Kidney",
            29663 : "Left Kidney"
        }

"""1247 : "Trachea",
            1302 : "Right Lung",
            1326 : "Left Lung",
            170 : "Pancreas",
            187 : "Gallbladder",
            237 : "Urinary Bladder",
            2473 : "Sternum",
            29193 : "First Lumbar Vertebra",
            29662 : "Right Kidney",
            29663 : "Left Kidney","""

okok = [i for i in organs]

cwd = os.getcwd()
start = time()

obj = os.scandir(cwd+'\\Patients3D\\' if '\\' in cwd else cwd+'/Patients3D/')
list_files = [i.name for i in obj]

for i in list_files:
    epi_img = nib.load(cwd+'\\Patients3D\\'+i)
    epi_img_data = epi_img.get_fdata()
    print(i)
    for x in range(1):
        images = np.array(epi_img_data[:, 275, :])
        for y in images:
            for z in range(epi_img_data.shape[2]):
                y[z] = translate(y[z])
        image = Image.fromarray(np.asarray(images, dtype='uint8'))
        filename = f"test{270}.png"
        #image = image.transpose(Image.ROTATE_90)
        image.save(cwd+'\\Imagettes3D\\'+filename if '\\' in cwd else cwd+'/Imagettes3D/'+filename)
        print(f"level {x} done", end='\r')
    obj = os.scandir(cwd+'\\centers\\' if '\\' in cwd else cwd+'/centers/')
    list_center_files = [a.name for a in obj if i.split('.')[0] in a.name]

    input_image = Image.open(cwd+"\\Imagettes3D\\test270.png")
    input_image = input_image.convert('RGB')
    draw = ImageDraw.Draw(input_image)
    colors = [  
                "red", "green", "blue", "yellow",
                "purple", "orange", "pink", "maroon",
                "white", "limegreen", "magenta", "lightsalmon",
                "mediumblue", "olivedrab", "peru", "sienna",
                "violet", "burlywood", "aqua", "goldenrod",
                "navy", "oldlace", "moccasin"
             ]
    a = input_image.size
    print(list_center_files)
    for a,color in zip(list_center_files,colors):
        if int(a.split('_')[4]) not in okok:
            continue
        f = open(cwd+'\\centers\\'+a if '\\' in cwd else cwd+'/centers/'+a)
        csvreader = csv.reader(f)
        organCoor = []
        for z in csvreader:
            organCoor = list(map(float,z))
        print(organCoor)
        draw.line([(organCoor[2]-6,organCoor[0]-6),(organCoor[2]+6,organCoor[0]+6)], width=5, 
            fill=color)
        draw.line([(organCoor[2]+6,organCoor[0]-6),(organCoor[2]-6,organCoor[0]+6)], width=5, 
            fill=color)
    input_image = input_image.transpose(Image.ROTATE_90)
    input_image.save(cwd+'\\real\\'+i+".png" if '\\' in cwd else cwd+'/real/'+i+".png")

"""print(images)

for i in images:
    for j in range(33**2):
        i[j] = translate(i[j])
print(images)
print(len(images.tolist()[0]))


images = images[0:15**2]
for i in images:
    image = i.reshape(33,33)
    image = Image.fromarray(np.asarray(image, dtype='uint8'))
    filename = f"test{lol}.png"
    image.save(cwd+'\\ImagettesTest\\'+filename if '\\' in cwd else cwd+'/ImagettesTest/'+filename)
    lol += 1"""

print(time()-start)