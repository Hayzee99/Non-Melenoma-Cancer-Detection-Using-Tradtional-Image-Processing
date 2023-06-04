import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

def segment_type(arg):
    
    keyList = [["GLD", [115,    0,     108]],
               ["INF", [122,    1,     145]],
               ["FOL", [148,    47,    216]],
               ["HYP", [242,    246,   254]],
               ["RET", [130,    9,     181]],
               ["PAP", [157,    85,    236]],
               ["EPI", [106,    0,     73]],
               ["KER", [168,    123,   248]],
               ["BKG", [0,      0,     0]],
               ["BCC", [255,    255,   127]],
               ["SCC", [142,    255,   127]],
               ["IEC", [127,    127,   255]],
               ]

    for i in range(len(keyList)):
        if (arg == keyList[i][1]).all():
            return keyList[i][0]

def texture_features(arg_img):
    glcm = graycomatrix(cv2.cvtColor(arg_img, cv2.COLOR_BGR2GRAY), [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm,prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')

    
    return contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]


def edge_mag(arg_img):
    arg_img = cv2.cvtColor(arg_img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(arg_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arg_img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.abs(np.sqrt(gx**2 + gy**2 ))
    mag = np.mean(mag)

    return mag

img_dir = "C:\\Users\\Haadin\\PycharmProjects\\DIP_Project\\Queensland Dataset CE42\\Training\\Images"
mask_dir = "C:\\Users\\Haadin\\PycharmProjects\\DIP_Project\\Queensland Dataset CE42\\Training\\Masks"

GLD_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

INF_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

FOL_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

HYP_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

RET_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

PAP_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

EPI_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

KER_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

BKG_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

BCC_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

SCC_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

IEC_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )


img_filenames = os.listdir(img_dir)
mask_filenames = os.listdir(mask_dir)
count = 0

for k in range(0,len(img_filenames), 3):
    print(count, " / 400")
    count +=1
    img_path = os.path.join(img_dir, img_filenames[k])
    mask_path = os.path.join(mask_dir, mask_filenames[k])

    # print(img_filenames[0:5])

    # print(mask_filenames[0:5])


    img = cv2.imread(img_path, 1)

    mask = cv2.imread(mask_path, 1)



    for i in range(0, 256, 16):
    
        for j in range(0, 256, 16):
            
            patch_img = img[i:i+16, j:j+16, :]
            tissue_type = segment_type(mask[i+7, j+7, :])
            
            contrast, dissimilarity, homogeneity, energy, correlation = texture_features(patch_img)
            entropy = shannon_entropy(patch_img)
            mean = np.mean(patch_img)
            variance = np.var(patch_img)
            sobel_mag = edge_mag(patch_img)
            
            new_row = {
                'contrast': contrast,
                'dissimilarity': dissimilarity,
                'homogenetiy' : homogeneity,
                'energy': energy,
                'correlation': correlation,
                'entropy' : entropy,
                'mean' : mean,
                'variance' : variance,
                'sobel_mag' : sobel_mag
                }
            
            match tissue_type:
                case "GLD":
                    GLD_df.loc[len(GLD_df)] = new_row

                case "INF":
                    INF_df.loc[len(INF_df)] = new_row

                case "FOL":
                    FOL_df.loc[len(FOL_df)] = new_row

                case "HYP":
                    HYP_df.loc[len(HYP_df)] = new_row

                case "RET":
                    RET_df.loc[len(RET_df)] = new_row

                case "PAP":
                    PAP_df.loc[len(PAP_df)] = new_row

                case "EPI":
                    EPI_df.loc[len(EPI_df)] = new_row

                case "KER":
                    KER_df.loc[len(KER_df)] = new_row

                case "BKG":
                    BKG_df.loc[len(BKG_df)] = new_row

                case "BCC":
                    BCC_df.loc[len(BCC_df)] = new_row

                case "SCC":
                    SCC_df.loc[len(SCC_df)] = new_row

                case "IEC":
                    IEC_df.loc[len(IEC_df)] = new_row

            

print(FOL_df)
        
                
print("\n -- GLD -- \n")
print(GLD_df.mean())
print("\n -- INF -- \n")
print(INF_df.mean())
print("\n -- FOL -- \n")
print(FOL_df.mean())
print("\n -- HYP -- \n")
print(HYP_df.mean())
print("\n -- RET -- \n")
print(RET_df.mean())
print("\n -- PAP -- \n")
print(PAP_df.mean())
print("\n -- EPI -- \n")
print(EPI_df.mean())
print("\n -- KER -- \n")
print(KER_df.mean())
print("\n -- BKG -- \n")
print(BKG_df.mean())
print("\n -- BCC -- \n")
print(BCC_df.mean())
print("\n -- SCC -- \n")
print(SCC_df.mean())
print("\n -- IEC -- \n")
print(IEC_df.mean())

mean_df = pd.DataFrame(columns=['contrast', 'dissimilarity', 'homogenetiy',
                                'energy', 'correlation', 'entropy', 'mean',
                                 'variance', 'sobel_mag' ] )

mean_df = pd.concat([GLD_df.mean(), INF_df.mean(), FOL_df.mean(), HYP_df.mean(),
                     RET_df.mean(), PAP_df.mean(), EPI_df.mean(), KER_df.mean(),
                     BKG_df.mean(), BCC_df.mean(), SCC_df.mean(), IEC_df.mean()], axis=1)

print("\n -- mean_df --\n")
print(mean_df)

contrast_row = mean_df.loc[['contrast']]

print(contrast_row)

mean_df.to_csv('feature_vector.csv',index=True, encoding='utf-8')










