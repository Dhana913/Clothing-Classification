import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['Pants', 'Shirt', 'Shoes', 'Skirt', 'T-Shirt']


# Load the pre-trained ResNet model
class ClothesClassifierResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except those that will be explicitly unfrozen later
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer 4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
    """
        Predicts the class of an image using a trained ResNet model.

        Args:
            image_path: The path to the image file.

        Returns:
            str: The predicted class name.
        """
    image = Image.open(image_path).convert("RGB")
    # Define image transformations and normalize tensor
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0) # Apply transformations and add a batch dimension

    global trained_model

    if trained_model is None: # Load the model if it's not already loaded
        trained_model = ClothesClassifierResNet()
        trained_model.load_state_dict(torch.load("model\saved_model.pth"))
        trained_model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculation for inference
        output = trained_model(image_tensor) # Get the model's output
        _, predicted_class = torch.max(output, 1) # Get the predicted class index
        return class_names[predicted_class.item()] # Return the predicted class name
