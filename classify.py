import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from PIL import Image
import gradio as gr



class  LungDisease_Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super( LungDisease_Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flattened_size = 64 * 24 * 24  

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
model =  LungDisease_Classifier(num_classes=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "C:/Users/Imtiaz/Documents/GitHub/Lung_Colon_Cancer_Classification_with_CNN/LungDisease_Classifier.pth" 

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

model.eval() 

transform = transforms.Compose([
    transforms.Resize((192, 192)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7290, 0.6000, 0.8762], std=[0.1752, 0.2094, 0.0970])
])


class_names = ['Colon Adenocarcinoma', 'Normal Colon Tissue', 'Lung Adenocarcinoma', 'Normal Lung Tissue', 'Lung Squamous Cell_Carcinoma'] 


def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(img) 
        probabilities = F.softmax(output, dim=1) 
        pred_idx = torch.argmax(probabilities, dim=1).item() 
        confidence = probabilities[0, pred_idx].item() 
    
    return {class_names[i]: float(probabilities[0, i]) for i in range(len(class_names))}


sample_images= ['colonca720.jpeg','colonn769.jpeg','lungaca805.jpeg','lungn923.jpeg','lungscc950.jpeg']


interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3), 
    examples=sample_images,
    title="Lung and Colon Cancer Classifier",
    description="Upload any histopathological image of lung or colon to classify among 5 classes."
)

interface.launch()