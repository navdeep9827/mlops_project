import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the function to predict sign language from an image
def predict_sign_language(image_path, model, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move image to device
    image = image.to(device)

    # Make prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = train_data.classes[predicted.item()]  # Assuming train_data is accessible here

    return predicted_label

# Load the trained model
model = ResNet9(3, target_num)  # Assuming `target_num` is defined in your code
model.load_state_dict(torch.load('ISN-2-custom-resnet.pth'))  # Load model weights
model.eval()  # Set model to evaluation mode

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage
image_path = 'path_to_your_image.jpg'  # Change this to the path of your image
predicted_label = predict_sign_language(image_path, model, device)
print('Predicted Sign Language:', predicted_label)
