from tqdm import tqdm
from data_loader import ClassOrNOT, TETS_DATA
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import os # Allows to access files
import numpy as np
import matplotlib.pyplot as plt
import time

img_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)), # Resize the image from whatever it is to a [3, 224, 224] image
        transforms.RandomHorizontalFlip(p=0.5), # Do a random flip with 50% probability
        transforms.ToTensor(), # Convert the image to a Tensor
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # Normalize the Image
    ]
)

def voc_eval_cls(file_name, ids, cls, confidence, draw=False):

    # Load test set
    gtids, gt = [], []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split()
            gtids.append(parts[0])
            gt.append(int(parts[1]))
    gt = np.array(gt)

    # Map results to ground truth
    out = np.ones(len(gt)) * -np.inf
    start_time = time.time()
    for i, img_id in enumerate(ids):
        if time.time() - start_time > 1:
            print(f'{cls}: pr: {i+1}/{len(ids)}')
            start_time = time.time()

        try:
            j = gtids.index(img_id)
        except ValueError:
            raise ValueError(f'unrecognized image "{img_id}"')

        out[j] = confidence[i]

    # Compute precision/recall
    sorted_indices = np.argsort(-out)
    tp = gt[sorted_indices] > 0
    fp = gt[sorted_indices] < 0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / np.sum(gt > 0)
    prec = tp_cum / (tp_cum + fp_cum)

    # Compute AP (11-point interpolation)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = prec[rec >= t]
        ap += (np.max(p) if p.size > 0 else 0)

    ap /= 11.0

    # Plot if needed
    if draw:
        plt.plot(rec, prec, '-')
        plt.grid(True)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'class: {cls}, subset: TESTING, AP = {ap:.3f}')
        plt.show()

    return rec, prec, ap

birdOrNot_train = ClassOrNOT("nndl_proj/data/train_set",
                     img_transforms, 'bird', 'not_bird')

birdOrNot_val = ClassOrNOT("nndl_proj/data/val_set",
                     img_transforms, 'bird', 'not_bird')


### SELECT DEVICE ###
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on Device {DEVICE}")

model = resnet18(pretrained=True) # Load the ResNet18 Model
num_ftrs = model.fc.in_features # Get the number of features in the last layer
model.fc = nn.Linear(num_ftrs, 1) # Change the last layer to have 2 outputs
model = model.to(DEVICE) # Send the model to the device

### Loss Function ###
loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss with Logits

### Optimizer ###
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam Optimizer with learning rate of 0.001
### DataLoader ###
train_loader = DataLoader(birdOrNot_train, batch_size=32, shuffle=True) # DataLoader for training set
val_loader = DataLoader(birdOrNot_val, batch_size=32, shuffle=False) # DataLoader for validation set

### Training Loop ###
num_epochs = 2 # Number of epochs to train for

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, train_meta, val_meta, cls):
    log_training = {"epoch": [],
                    "training_loss": [],
                    "training_acc": [],
                    "validation_loss": [],
                    "validation_acc": []}
    for epoch in range(1, num_epochs + 1):
            print(f"Starting Epoch {epoch}")
            training_losses = []
            validation_losses = []
            
            model.train() # Turn On BatchNorm and Dropout
            ids_train = []
            confidances_train = []
            for image, label, file_names in tqdm(train_loader):
                image, label = image.to(DEVICE), label.to(DEVICE).float() # Send the image and label to the device
                label = label.view(-1, 1) # Reshape the label to be [batch_size, 1]

                optimizer.zero_grad()
                out = model.forward(image)
            
                ### CALCULATE LOSS ##
                loss = loss_fn(out, label)
                training_losses.append(loss.item())

                ### CALCULATE ACCURACY ###
                predictions = torch.sigmoid(out)
                pred_2 = predictions.view(-1).detach().cpu().numpy()
                ids_train.extend(file_names)
                confidances_train.extend(pred_2)

                loss.backward()
                optimizer.step()

            model.eval() # Turn Off Batchnorm
            ids_vals = []
            confidances_vals = []

            for image, label, file_names in tqdm(val_loader):
                image, label = image.to(DEVICE), label.to(DEVICE).float()
                with torch.no_grad():
                    out = model.forward(image)
                    label = label.view(-1, 1)
                    ### CALCULATE LOSS ##
                    loss = loss_fn(out, label)
                    validation_losses.append(loss.item())

                    ### CALCULATE ACCURACY ###
                    predictions = torch.sigmoid(out)
                    pred_2 = predictions.view(-1).cpu().numpy()
                    ids_vals.extend(file_names)
                    confidances_vals.extend(pred_2)


            training_loss_mean = np.mean(training_losses)
            valid_loss_mean = np.mean(validation_losses)

            _, _, ap_train = voc_eval_cls(train_meta, ids_train, cls, confidances_train, draw=False)
            _, _, ap_val = voc_eval_cls(val_meta, ids_vals, cls, confidances_vals, draw=False)

            log_training["epoch"].append(epoch)
            log_training["training_loss"].append(training_loss_mean)
            log_training["training_acc"].append(ap_train)
            log_training["validation_loss"].append(valid_loss_mean)
            log_training["validation_acc"].append(ap_val)

            print("Training Loss:", training_loss_mean) 
            print("Training AP:", ap_train)
            print("Validation Loss:", valid_loss_mean)
            print("Validation AP:", ap_val)
        
#functio that takes a trained model, test data, predicts and save txt file with format image_name probablity
def predict_and_save(model, test_loader, output_file):
    model.eval()
    with torch.no_grad():
        with open(output_file, 'w') as f:
            for images, file_names in tqdm(test_loader):
                images = images.to(DEVICE)
                out = model(images)
                predictions = torch.sigmoid(out)
                predictions = predictions.view(-1).cpu().numpy()

                for file_name, prob in zip(file_names, predictions):
                    #pred to 6 decimal places
                    prob = f"{prob:.6f}"
                    f.write(f"{file_name} {prob}\n")

    print(f"Predictions saved to {output_file}")


# Load the test set
# birdOrNot_test = TETS_DATA("nndl_proj/data/test_set",
#                      img_transforms)

# # Create DataLoader for test set
# test_loader = DataLoader(birdOrNot_test, batch_size=32, shuffle=False)

# predict_and_save(model, test_loader, "nndl_proj/submissions/bird_test.txt")
