import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import requests
import torch.nn.functional as F

# --- 1. Model and Data Loading ---

@st.cache_resource
def load_model_and_labels():
    """
    Loads the pre-trained ResNet-18 model and ImageNet labels.
    Using @st.cache_resource ensures the model is loaded only once.
    """
    # Set up the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically import the model to avoid errors if torchvision is not installed
    from torchvision.models import resnet18, ResNet18_Weights

    # Load pre-trained ResNet-18 model and its weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # The preprocessing steps required for the model
    preprocess = weights.transforms()

    # Get the ImageNet class labels from a standard URL
    labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels_file = requests.get(labels_url).text.split('\n')
    
    return model, preprocess, labels_file, device

try:
    model, preprocess, labels_file, device = load_model_and_labels()
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop() # Stop the app if the model can't be loaded

# --- 2. Core Logic Functions ---

def get_class_name(class_id):
    """Returns the class name for a given class_id."""
    return labels_file[class_id]

def apply_aggressors(image, brightness, rotation, noise_level):
    """
    Applies a series of transformations (aggressors) to the input image.
    This function is the core of the validation test.
    """
    transform_pipeline = T.Compose([
        T.ColorJitter(brightness=brightness),
        T.RandomRotation(degrees=(rotation, rotation)),
        T.ToTensor(),
        T.Lambda(lambda x: x + torch.randn_like(x) * noise_level),
        T.Lambda(lambda x: torch.clamp(x, 0, 1)),
        T.ToPILImage()
    ])
    return transform_pipeline(image)

def predict(image):
    """
    Runs inference on the model and returns the top prediction and confidence.
    """
    img_t = preprocess(image).to(device)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    probabilities = F.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    class_name = get_class_name(top_catid.item())
    confidence = top_prob.item()
    
    return class_name, confidence

# --- 3. Streamlit Dashboard ---

st.set_page_config(layout="wide", page_title="ML Model Stress Test")
st.title("🤖 Adversarial Image Classifier Stress Test")
st.write("Upload an image and use the sliders to apply 'aggressors' (distortions). See how the model's prediction and confidence change in real-time. This project aims at finding a model's breaking points.")
st.markdown("---")

# --- Sidebar for user controls ---
st.sidebar.header("⚙️ Aggressor Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

brightness_val = st.sidebar.slider("Brightness", 0.2, 1.8, 1.0, 0.05)
rotation_val = st.sidebar.slider("Rotation (degrees)", 0, 45, 0, 1)
noise_val = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.0, 0.01)

# Main app layout
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(original_image, caption="Your uploaded image.", use_container_width=True)
        original_class, original_conf = predict(original_image)
        st.success(f"Prediction: **{original_class}**")
        st.info(f"Confidence: **{original_conf:.2%}**")

    with col2:
        st.header("Transformed Image")
        transformed_image = apply_aggressors(original_image, brightness_val, rotation_val, noise_val)
        st.image(transformed_image, caption="Image with aggressors applied.", use_container_width=True)
        new_class, new_conf = predict(transformed_image)

        if new_class != original_class:
            st.error(f"Prediction: **{new_class}** (Prediction Changed!)")
        else:
            st.success(f"Prediction: **{new_class}**")
            
        if new_conf < original_conf * 0.8:
             st.warning(f"Confidence: **{new_conf:.2%}** (Confidence Dropped)")
        else:
             st.info(f"Confidence: **{new_conf:.2%}**")
else:
    st.info("Awaiting image upload to begin analysis.")
