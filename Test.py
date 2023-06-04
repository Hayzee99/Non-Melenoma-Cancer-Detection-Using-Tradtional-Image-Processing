import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import metrics
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

def segment_BGR_code(arg):
    
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
        if (arg == keyList[i][0]):
            return keyList[i][1]

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

def test_contrast(val):
    idx = feature_df['contrast'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]


def test_dissimilarity(val):
    idx = feature_df['dissimilarity'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_homogenetiy(val):
    idx = feature_df['homogenetiy'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_energy(val):
    idx = feature_df['energy'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_correlation(val):
    idx = feature_df['correlation'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_entropy(val):
    idx = feature_df['entropy'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_mean(val):
    idx = feature_df['mean'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_variance(val):
    idx = feature_df['variance'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]
    
def test_sobel_mag(val):
    idx = feature_df['sobel_mag'].sub(val).abs().idxmin()

    #added double [[]] for one row DataFrame
    df1 = feature_df.loc[[idx]]

    index = df1.index
    return index[0]

def most_frequent(List):
    return max(set(List), key = List.count)

def classify(x1:int, x2, x3):
    if max([x1, x2, x3]) == x1:
        return 0
    elif max([x1, x2, x3]) == x2:
        return 1
    elif max([x1, x2, x3]) == x3:
        return 2
    else:
        return 0   

def fScore(actual, predicted) :
    bcc_tp = 1
    bcc_fp = 0
    bcc_fn = 0

    iec_tp = 1
    iec_fp = 0
    iec_fn = 0

    scc_tp = 1
    scc_fp = 0
    scc_fn = 0

    for i in range(100):
        if actual[i, 0] == 1 and predicted[i, 0] == 1:
            bcc_tp += 1
        elif actual[i, 0] == 0 and predicted[i, 0] == 1:
            bcc_fp += 1
        elif actual[i, 0] == 1 and predicted[i, 0] == 0:
            bcc_fn += 1

        if actual[i + 100, 1] == 1 and predicted[i + 100, 1] == 1:
            iec_tp += 1
        elif actual[i + 100, 1] == 0 and predicted[i + 100, 1] == 1:
            iec_fp += 1
        elif actual[i + 100, 1] == 1 and predicted[i + 100, 1] == 0:
            iec_fn += 1

        if actual[i + 200, 2] == 1 and predicted[i + 200, 2] == 1:
            scc_tp += 1
        elif actual[i + 200, 2] == 0 and predicted[i + 200, 2] == 1:
            scc_fp += 1
        elif actual[i + 200, 2] == 1 and predicted[i + 200, 2] == 0:
            scc_fn += 1

        print("bcc tp : ", bcc_tp)
        print("iec tp : ", iec_tp)
        print("scc tp : ", scc_tp)

    bcc_fScore = bcc_tp / bcc_tp + (0.5 * (bcc_fp + bcc_fn))
    iec_fScore = iec_tp / iec_tp + (0.5 * (iec_fp + iec_fn))
    scc_fScore = scc_tp / scc_tp + (0.5 * (scc_fp + scc_fn))

    return bcc_fScore, iec_fScore, scc_fScore

        

feature_df = pd.read_csv('C:\\Users\\Haadin\\PycharmProjects\\feature_vector.csv',index_col=[0])
img_dir = "C:\\Users\\Haadin\\PycharmProjects\\DIP_Project\\Queensland Dataset CE42\\Testing\\Images"


feature_df.index = ['contrast', 'dissimilarity', 'homogenetiy',
                    'energy', 'correlation', 'entropy', 'mean',
                    'variance', 'sobel_mag' ]
feature_df.columns =['GLD', 'INF', 'FOL', 'HYP', 'RET', 'PAP', 'EPI',
                     'KER', 'BKG', 'BCC', 'SCC', 'IEC' ] 
feature_df = feature_df.transpose()
print(feature_df)

imgs = np.load("C:\\Users\\Haadin\\PycharmProjects\\DIP_Project_img.npy")
true_labels = np.load("C:\\Users\\Haadin\\PycharmProjects\\DIP_Project_labels.npy")
assigned_labels = np.zeros([300, 3], dtype=np.uint8)
generated_masks = np.zeros([300, 256, 256, 3], dtype=np.uint8)

# print("LABEL : ", true_labels[150,:])
# cv2.imshow("test", imgs[150,...])

# cv2.waitKey(0)

mask = np.zeros([256, 256, 3], dtype=np.uint8)

for k in range(0, 300, 2):
    print(k)
    bcc_count = 0
    iec_count = 0
    scc_count = 0
    for i in range(0, 256, 8):

        for j in range(0, 256, 8):

            patch_img = imgs[k, i:i+16, j:j+16, :]
            
            contrast, dissimilarity, homogeneity, energy, correlation = texture_features(patch_img)
            entropy = shannon_entropy(patch_img)
            mean = np.mean(patch_img)
            variance = np.var(patch_img)
            sobel_mag = edge_mag(patch_img)

            featureVec = []
            featureVec.append(test_contrast(contrast))
            featureVec.append(test_dissimilarity(dissimilarity))
            featureVec.append(test_homogenetiy(homogeneity))
            featureVec.append(test_energy(energy))
            featureVec.append(test_correlation(correlation))
            featureVec.append(test_entropy(entropy))
            featureVec.append(test_mean(mean))
            featureVec.append(test_variance(variance))
            featureVec.append(test_sobel_mag(sobel_mag))

            seg_type = most_frequent(featureVec)
            mask[i:i+8, j:j+8, :] = segment_BGR_code(seg_type)

            match seg_type:
                case "BCC":
                    bcc_count += 1
                case "IEC":
                    iec_count += 1
                case "SCC":
                    scc_count += 1

    generated_masks[k,...] = mask
    assigned_labels[k, classify(bcc_count, iec_count, scc_count)] = 1

    print("classify = ", classify(bcc_count, iec_count, scc_count))

np.save("DIP_Project_generated_masks.npy", generated_masks)
np.save("DIP_Project_assigned_labels.npy", assigned_labels)



bcc_fScore, iec_fScore, scc_fScore = fScore(true_labels, assigned_labels)

cm_display = metrics.ConfusionMatrixDisplay((metrics.confusion_matrix(list(np.argmax(true_labels, axis=1)), list(np.argmax(assigned_labels, axis=1)))), display_labels=["BCC", "IEC", "SCC"])

print("\n\n -- F-Scores -- \n")
print("  BCC = ", bcc_fScore)
print("  IEC = ", iec_fScore)
print("  SCC = ", scc_fScore)
print("\n ---------------\n\n")

cm_display.plot()
plt.savefig("confusion_Mat.png")
plt.show()




    















