import os
import csv
train_dir = './deal-data/test' # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
with open('deal-data/test/test_lable.csv', 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label':label})
        label += 1
    train_csv.close()
