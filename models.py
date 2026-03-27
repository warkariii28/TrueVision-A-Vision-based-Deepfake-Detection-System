from app import db
from flask_login import UserMixin
import torch
import torch.nn as nn
import os
#from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
from sqlalchemy.orm import relationship
import traceback
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from gradcam_explainer import  get_gradcam_explanation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# User model for database
class User(UserMixin, db.Model):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pwd = db.Column(db.String(300), nullable=False)
    
    results = relationship('Result', backref='user', lazy=True)  # Relationship with Result table

    def __repr__(self):
        return f'<User {self.username}>'

# Result model for storing prediction results
class Result(db.Model):
    __tablename__ = "result"
    
    result_id = db.Column(db.Integer, primary_key=True)
    confidence_score = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    feedback = db.Column(db.String(200), nullable=True)  # Optional feedback
    image_path = db.Column(db.String(255), nullable=False)
    gradcam_path = db.Column(db.String(255), nullable=True)  # Path to saved gradcam image
    explanation = db.Column(db.Text, nullable=True)  # Explanation text from GradCAM analyzer
    recommendation = db.Column(db.Text, nullable=True)  # Recommendation based on analysis
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to User table
    
    def __repr__(self):
        return f"<Result {self.result_id} - Prediction: {self.prediction}>"



class Performance(db.Model):
    __tablename__ = 'performance'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    fpr = db.Column(db.Float)
    fnr = db.Column(db.Float)
    tnr = db.Column(db.Float)
    tp = db.Column(db.Integer)
    tn = db.Column(db.Integer)
    fp = db.Column(db.Integer)
    fn = db.Column(db.Integer)
    auc_roc = db.Column(db.Float)
    pr_auc = db.Column(db.Float)
    confusion_matrix = db.Column(db.Text, nullable=True)  # Add this line

# Custom ViT model without Grad-CAM
# Custom ViT model without Grad-CAM
class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()
        #model_name = "google/vit-base-patch16-224"
        model_name = "microsoft/swin-base-patch4-window7-224"
        #self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        #self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 2)
        )
        try:
            weights_path = os.path.join(BASE_DIR, "best_Swin_stage2.pth")
            path = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(path, strict=False)
            print("Model weights loaded successfully.")
            # Print the names and shapes of all tensors
            print("Tensors and their shapes in the .pth file:")
            for key, value in path.items():
                print(f"{key}: {value.shape}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    def forward(self, x):
        # Forward pass through the model
        outputs = self.model(x)
        return outputs.logits
    
    def predict(self, image):
        self.eval()
        # Process image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        print("Transformed input shape:", inputs['pixel_values'].shape)
        
        # No gradient tracking for inference
        with torch.no_grad():
            outputs = self.forward(inputs['pixel_values'])
        
        return outputs

# Load model and weights
model = CustomViT()
# model_path = "C:/Users/athar/Downloads/User-Authentication-in-Flask-main/User-Authentication-in-Flask-main/best_ViT_stage2.pth"
# try:
#     # Load state dict
#     state_dict = torch.load(model_path, map_location='cpu')
#     model.model.load_state_dict(state_dict)
#     print("Model weights loaded successfully.")
# except Exception as e:
#     print(f"Error loading model weights: {e}")
    
model.eval()

# Main prediction function
# Modified predict_image function to include GradCAM explanation
# Main prediction function with improved Grad-CAM
def predict_image(filepath):
    try:
        # Open and process image
        image = Image.open(filepath).convert("RGB")
        
        print(f"Original image size: {image.size}")  # Debug: Check image size
        
        # Get model prediction
        outputs = model.predict(image)
        print(f"Raw model outputs (logits): {outputs}")  # Debug: Check raw outputs
        
        # Get confidence scores and predicted class
        probs = torch.softmax(outputs, dim=1)
        print(f"Softmax probabilities: {probs}")  # Debug: Check softmax probabilities
        
        confidence, predicted_class = torch.max(probs, dim=1)
        
        # Map class indices to labels
        label_map = {0: "Fake", 1: "Real"}
        prediction = label_map.get(predicted_class.item(), "Unknown")
        
        print(f"Prediction: {prediction}, Confidence: {confidence.item()}")  # Debug: Check final prediction
        
        # Generate improved Grad-CAM and get image path
        gradcam_path, cam_array = generate_gradcam(filepath, prediction, confidence.item())
        
        # Get explanation based on the improved Grad-CAM
        is_fake = (prediction == "Fake")
        
        # Import get_gradcam_explanation here to avoid circular imports
        from gradcam_explainer import get_gradcam_explanation
        
        gradcam_explanation = get_gradcam_explanation(
            cam_array, 
            image.size, 
            confidence.item(),
            is_fake=is_fake
        )
        
        # Prepare relative paths for frontend
        filename = os.path.basename(filepath)
        image_rel_path = os.path.join("uploads", filename).replace("\\", "/")
        gradcam_rel_path = os.path.join("gradcam", os.path.basename(gradcam_path)).replace("\\", "/")
        
        # Return results with explanation
        return {
            "prediction": prediction,
            "confidence": round(confidence.item() * 100, 4),
            "image_path": image_rel_path,
            "gradcam_path": gradcam_rel_path,
            "explanation": gradcam_explanation["explanation"],
            "recommendation": gradcam_explanation["recommendation"]
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return {"error": f"Error during prediction: {str(e)}"}
    
# Hook storage
activations = None
grads = None

# Hook functions
def save_activations(module, input, output):
    global activations
    activations = output.detach()  # shape: [1, num_patches+1, hidden_dim]

def save_gradients(module, grad_input, grad_output):
    global grads
    grads = grad_output[0].detach()  # shape: [1, num_patches+1, hidden_dim]

# Register hooks to the transformer encoder block
try:
    #target_block = model.model.vit.encoder.layer[-1].output # Last Normalization block
    target_block = model.model.swin.encoder.layers[-1].blocks[-1].output.dense
    target_block.register_forward_hook(save_activations)
    target_block.register_full_backward_hook(save_gradients)
except Exception as e:
    print(f"Hook registration failed: {e}")

# Enhanced Grad-CAM implementation with dynamic highlighting
def generate_gradcam(image_path, prediction, confidence):
    global activations, grads
    try:
        model.eval()
        image = Image.open(image_path).convert("RGB")
        # Preprocess
        inputs = model.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        pixel_values.requires_grad = True
    
        # Forward pass
        outputs = model.model(pixel_values).logits
        pred_class = outputs.argmax(dim=1)
        
        # Backward pass to get gradients
        one_hot = torch.zeros_like(outputs)
        one_hot[0, pred_class] = 1
        outputs.backward(gradient=one_hot)
    
        if activations is None or grads is None:
            raise ValueError("Grad-CAM hooks did not capture data properly.")
        
        # Grad-CAM calculations with improvements
        # Average gradients over the sequence dimension (patches)
        pooled_grads = grads.mean(dim=1)  # [1, C]
        
        # Weight activations by pooled gradients
        weights = pooled_grads[0]  # [C]
        cam = torch.matmul(activations[0], weights)  # [196]
        
        # Reshape to 7×7 patch grid
        cam = cam.reshape(7, 7).cpu().numpy()
        
        # Dynamic thresholding based on prediction type and confidence
        is_fake = (prediction == "Fake")
        
        if is_fake:
            # For fake images: highlight top suspicious regions
            # Higher confidence fake = more aggressive highlighting
            if confidence > 0.95:
                percentile_value = np.percentile(cam, 60)  # Top 40% regions
                power_factor = 2.0  # Strong contrast
            elif confidence > 0.85:
                percentile_value = np.percentile(cam, 70)  # Top 30% regions
                power_factor = 1.8
            else:
                percentile_value = np.percentile(cam, 75)  # Top 25% regions
                power_factor = 1.5
        else:
            # For real images: highlight uncertain/suspicious regions
            # Lower confidence real = more regions to highlight
            uncertainty = 1 - confidence
            if uncertainty > 0.3:  # Very uncertain
                percentile_value = np.percentile(cam, 65)  # Top 35% regions
                power_factor = 1.8
            elif uncertainty > 0.15:  # Moderately uncertain
                percentile_value = np.percentile(cam, 75)  # Top 25% regions
                power_factor = 1.6
            else:  # Confident real
                percentile_value = np.percentile(cam, 85)  # Top 15% regions
                power_factor = 1.4
        
        # Apply thresholding to focus on most important regions
        cam = np.maximum(cam - percentile_value, 0)
        
        # Apply non-linear scaling to increase contrast
        cam = np.power(cam, power_factor)
        
        # Normalize only if we have non-zero values
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Store the normalized CAM for explainability
        cam_array = cam.copy()
        
        # Resize CAM to original image size
        cam = cv2.resize(cam, image.size)
        
        # Choose colormap based on prediction
        if is_fake:
            # Red-orange colormap for fake (suspicious regions)
            colormap = cv2.COLORMAP_HOT
        else:
            # Blue-green colormap for real (uncertain regions)
            colormap = cv2.COLORMAP_COOL
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        
        # Convert PIL image to numpy for blending
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV compatibility
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Create overlay with dynamic alpha blending
        if is_fake and confidence > 0.9:
            # Strong overlay for high-confidence fake
            overlay = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)
        elif is_fake:
            # Medium overlay for lower-confidence fake
            overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
        else:
            # Subtle overlay for real images
            overlay = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)
        
        # Convert back to RGB for PIL
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Save or display result
        overlay_img = Image.fromarray(overlay)
        base_name = os.path.splitext(image_path)[0]
        
        # Fix this path to save in static/gradcam/
        overlay_path = os.path.join("static", "gradcam", os.path.basename(base_name) + "_gradcam.jpg")
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        overlay_img.save(overlay_path)
        print(f"Grad-CAM image saved at: {overlay_path}")
        
        # Return both the path and the raw CAM array for explainability
        return overlay_path, cam_array
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        traceback.print_exc()
        return None, None
