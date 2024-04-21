# get the validation data
import torch

from first_implementation import load_model
from first_implementation.cnn_class import cnn_architecture

val_data = load_model.val_data
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# load the model
model = cnn_architecture().to(device)
model.load_state_dict(torch.load('model.pth'))

def predict_output(img):
    loader = CustomImageDataset(img)
    loader = DataLoader(loader, batch_size=20, shuffle=False)
    pred = test(loader, model)
    pred_label = pred.argmax(dim=1).to("cpu").numpy()
    return pred_label[0]