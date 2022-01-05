# NYCU_VRDL_FINAL_RESULTS
results for kaggle rsna pneumonia detection challenge using yolov5, fasterrcnn, retinanet

- submission.csv: fasterrcnn
- submission(1).csv: yolov5
- answer.csv: retinanet
- submission_ensemble4_v34.csv: result after ensumbling

## Reproduce the final result
1. train on [yolov5](https://github.com/ultralytics/yolov5)
2. produce submission file by make_submission.py
3. ensumble results using ensembleToCSV.py
