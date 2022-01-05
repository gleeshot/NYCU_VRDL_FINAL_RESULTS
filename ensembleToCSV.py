# ensembling the ensemble

# same with listmaker, but this one I plan to ensemble Egor's submission file

import ensemble
import pandas as pd

submissionfilename = 'submission_ensemble_square_v6.csv'
originalcsvpath = 'stage_2_sample_submission_original.csv'

ensemble1 = 'submission_ensemble2_es_v10.csv'
ensemble2 = "submission_ensemble4_v25.csv"
# ensemble3 = "submission_v3_dexp32_320.csv"
# ensemble4 = "submission_v3_dexp34_320.csv"
# ensemble5 = "submission(1).csv"
# ensemble6 = "submission_v3_dexp27.csv"

df = pd.read_csv(originalcsvpath)
weights = [0.09, 5]
#[0.5, 0.5] - ensemble2_ensemble_square_v1 confad 0.05 ensembleIOU = 0.4 (no bbox adjust)
#[0.5, 0.5] - ensemble2_ensemble_square_v2 confad 0.04 ensembleIOU = 0.4 (no bbox adjust)
#[0.1, 3] - ensemble2_ensemble_square_v3 confad 0.04 ensembleIOU = 0.4 (no bbox adjust)
#[0.09, 5] - ensemble2_ensemble_square_v4 confad 0.04 ensembleIOU = 0.4 (no bbox adjust)
#[0.09, 5] - ensemble2_ensemble_square_v5 confad 0.04 ensembleIOU = 0.1 (no bbox adjust)
num_model = 2
confadjust = 0.04  # after ensemble confidence
data1 = pd.read_csv(ensemble1, index_col ="patientId")
data2= pd.read_csv(ensemble2, index_col ="patientId")
# data3 = pd.read_csv(ensemble3, index_col ="patientId")
# data4 = pd.read_csv(ensemble4, index_col ="patientId")
# data5 = pd.read_csv(ensemble5, index_col ="patientId")
#data6 = pd.read_csv(ensemble6, index_col ="patientId")
def getimglistodet(description1):       # create list of detection per model
    if pd.isnull(description1) == False:
        s = description1.split(' ') # convert string into a list of numbers
        s2 = [float(x) for x in s[:-1]] # omit last empty value
        lens2 = int(len(s2)/5)

        listodet = []
        for j in range(lens2):
            conf, x, y, w, h = s2[j*5:(j*5)+5]
            cls = '0'
            x = x + (w/2)
            y = y + (h/2)
            det = [x,y,w,h,cls,conf]
            listodet.append(det)

    elif pd.isnull(description1):
        listodet = []

    return listodet

# [box_x, box_y, box_w, box_h, class, confidence]

# initialize submission file
submission_file = open(submissionfilename, 'w')
submission_file.write('patientId,PredictionString\n')

for i in range(len(df)):
    imgname = df.loc[i, 'patientId'] # get from original csv
    description1 = data1.loc[imgname]['PredictionString']
    description2 = data2.loc[imgname]['PredictionString']
    # description3 = data3.loc[imgname]['PredictionString']
    # description4 = data4.loc[imgname]['PredictionString']
    # description5 = data5.loc[imgname]['PredictionString']
    #description6 = data6.loc[imgname]['PredictionString']

    multimodel = []

    multimodel.append(getimglistodet(description1))
    multimodel.append(getimglistodet(description2))
    # multimodel.append(getimglistodet(description3))
    # multimodel.append(getimglistodet(description4))
    # multimodel.append(getimglistodet(description5))
    #multimodel.append(getimglistodet(description6))

    #print((multimodel))
    #get ensemble

    newdata = ensemble.GeneralEnsemble(multimodel, 0.1, weights=weights)

    coords = ""
    for line in newdata:
        x,y,w,h,cls,conf = line
        x = x - (w/2)
        y = y - (h/2)

        # decrease box size by ___%
        # x = x + (w*0.10)
        # y = y + (h*0.10)
        # w = w*0.85
        # h = h*0.85


        if conf > confadjust:
            coords +=(f'{conf} {x} {y} {w} {h} ')
        # line for multiple dets

    newline = f'{imgname},{coords}\n'
    submission_file.write(newline)
    # [box_x, box_y, box_w, box_h, class, confidence]
    # newline = [conf, x,y,w,h]

    print(i)
    #write to new csv file
