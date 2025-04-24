#DATA AND MODEL SETUP AND TRAINING
from torchvision import transforms
from torchvision.models import resnet18
from data_loader import ClassOrNOT, TETS_DATA
from torch.utils.data import DataLoader
from train import train_model, predict_and_save
import torch
import torch.nn as nn

img_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)), # Resize the image from whatever it is to a [3, 224, 224] image
        transforms.RandomHorizontalFlip(p=0.5), # Do a random flip with 50% probability
        transforms.ToTensor(), # Convert the image to a Tensor
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # Normalize the Image
    ]
)

birdOrNot_train = ClassOrNOT("nndl_proj/data/train_set",
                     img_transforms, 'bird', 'not_bird')

birdOrNot_val = ClassOrNOT("nndl_proj/data/val_set",
                     img_transforms, 'bird', 'not_bird')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on Device {DEVICE}")

model = resnet18(pretrained=True) # Load the ResNet18 Model
num_ftrs = model.fc.in_features # Get the number of features in the last layer
model.fc = nn.Linear(num_ftrs, 1) # Change the last layer to have 2 outputs
model = model.to(DEVICE) # Send the model to the device

loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss with Logits

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(birdOrNot_train, batch_size=32, shuffle=True)
val_loader = DataLoader(birdOrNot_val, batch_size=32, shuffle=False)

num_epochs = 2

train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, "nndl_proj/meta_datas/bird_train.txt", "nndl_proj/meta_datas/bird_val.txt", 'bird')

#Load the test set
birdOrNot_test = TETS_DATA("nndl_proj/data/test_set",
                     img_transforms)

# Create DataLoader for test set
test_loader = DataLoader(birdOrNot_test, batch_size=32, shuffle=False)

predict_and_save(model, test_loader, "nndl_proj/submissions/bird_test.txt")
