from pathlib import Path
import csv 


def convert_coordinates(label_parts):
    x_center = float(label_parts[1])
    y_center = float(label_parts[2])
    width = float(label_parts[3])
    height = float(label_parts[4])

    x_min = (x_center - width/2) * w_img
    y_min = (y_center - height/2) * h_img
    width = width * w_img
    height = height * h_img

    return x_min, y_min, width, height

w_img, h_img = 1024, 1024
experiment = 'exp35'
experiment_dir = Path(f'../../yolov5/runs/detect/{experiment}/labels')
test_images_dir = Path('../data/testing/images')
predicted_images = []
all_images = []
submission_labels = []

for label in experiment_dir.iterdir():
    predicted_images.append(label.stem)

for image in test_images_dir.iterdir():
    all_images.append(image.stem)

images_without_predictions = set(predicted_images).symmetric_difference(set(all_images))
for label in experiment_dir.iterdir():
    with open(label, 'r') as label_file:
        image_id = label.stem
        submission_label = ''

        for line in label_file:
            label_parts = line.strip().split(' ')
            x, y, width, height = convert_coordinates(label_parts)
            confidence = label_parts[5]
            submission_label += (f'{confidence} {x} {y} {width} {height} ')
        submission_label = f'{image_id},{submission_label}\n'
        submission_labels.append(submission_label)
    

with open('submission_v3_dexp35_320.csv', 'w') as submission_file:
    submission_file.write('patientId,PredictionString\n')
    submission_file.writelines(submission_labels)

with open('submission_v3_dexp35_320.csv', 'a') as submission_file:
    for image in images_without_predictions:
        submission_file.write(f'{image},\n')


