# NYCU_VRDL_FINAL_RESULTS
results for kaggle rsna pneumonia detection challenge we had tried using yolov5, fasterrcnn, retinanet

- submission.csv: fasterrcnn
- submission(1).csv: yolov5
- answer.csv: retinanet
- submission_ensemble4_v34.csv: result after ensumbling

## Reproduce the final result
1. get weight in [google drive](https://drive.google.com/drive/folders/13zyBCJ__VjSwHMfXxFUNJQ_4bXtHaORS)
2. train on [yolov5](https://github.com/ultralytics/yolov5)
3. produce submission file by make_submission.py
4. ensumble results using ensembleToCSV.py
