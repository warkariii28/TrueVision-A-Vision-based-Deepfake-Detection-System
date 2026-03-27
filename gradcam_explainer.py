import numpy as np
from typing import Dict, Tuple

def analyze_gradcam_regions(cam_array: np.ndarray) -> Dict:
    """
    Analyze the Grad-CAM heatmap to identify key regions and patterns.
    
    Args:
        cam_array: Normalized Grad-CAM activation map (7x7)
    
    Returns:
        Dictionary containing analysis results
    """
    if cam_array is None or cam_array.size == 0:
        return {
            "num_hot_regions": 0,
            "max_activation": 0,
            "avg_activation": 0,
            "center_focus": False,
            "edge_focus": False,
            "distribution": "uniform"
        }
    
    # Calculate basic statistics
    max_activation = np.max(cam_array)
    avg_activation = np.mean(cam_array)
    num_hot_regions = np.sum(cam_array > 0.5)  # Regions with >50% activation
    
    # Analyze spatial distribution
    center_region = cam_array[2:5, 2:5]  # Central 3x3 region
    edge_regions = np.concatenate([
        cam_array[0, :].flatten(),  # Top edge
        cam_array[-1, :].flatten(),  # Bottom edge
        cam_array[:, 0].flatten(),  # Left edge
        cam_array[:, -1].flatten()   # Right edge
    ])
    
    center_activation = np.mean(center_region)
    edge_activation = np.mean(edge_regions)
    
    # Determine focus patterns
    center_focus = center_activation > edge_activation * 1.5
    edge_focus = edge_activation > center_activation * 1.5
    
    # Determine distribution pattern
    if num_hot_regions <= 2:
        distribution = "concentrated"
    elif num_hot_regions >= 6:
        distribution = "dispersed"
    else:
        distribution = "moderate"
    
    return {
        "num_hot_regions": int(num_hot_regions),
        "max_activation": float(max_activation),
        "avg_activation": float(avg_activation),
        "center_focus": center_focus,
        "edge_focus": edge_focus,
        "distribution": distribution,
        "center_activation": float(center_activation),
        "edge_activation": float(edge_activation)
    }

def get_gradcam_explanation(cam_array: np.ndarray, image_size: Tuple[int, int], 
                          confidence: float, is_fake: bool) -> Dict[str, str]:
    """
    Generate dynamic explanation based on Grad-CAM analysis.
    
    Args:
        cam_array: Normalized Grad-CAM activation map
        image_size: Original image dimensions (width, height)
        confidence: Model confidence score (0-1)
        is_fake: Whether the prediction is fake
    
    Returns:
        Dictionary with explanation and recommendation
    """
    # Analyze the Grad-CAM regions
    analysis = analyze_gradcam_regions(cam_array)
    
    # Generate explanation based on prediction type
    if is_fake:
        explanation = generate_fake_explanation(analysis, confidence)
        recommendation = generate_fake_recommendation(analysis, confidence)
    else:
        explanation = generate_real_explanation(analysis, confidence)
        recommendation = generate_real_recommendation(analysis, confidence)
    
    return {
        "explanation": explanation,
        "recommendation": recommendation,
        "analysis": analysis  # Include raw analysis for debugging
    }

def generate_fake_explanation(analysis: Dict, confidence: float) -> str:
    """Generate explanation for fake image predictions."""
    base_explanation = "The model detected unusual attention patterns around the "
    
    # Determine focus areas
    focus_areas = []
    if analysis["center_focus"]:
        focus_areas.append("facial features")
    if analysis["edge_focus"]:
        focus_areas.append("image boundaries")
    
    if not focus_areas:
        if analysis["distribution"] == "dispersed":
            focus_areas.append("multiple regions across the face")
        else:
            focus_areas.append("specific facial areas")
    
    focus_description = " and ".join(focus_areas)
    
    # Build explanation based on confidence and analysis
    if confidence > 0.95:
        explanation = f"{base_explanation}{focus_description}, which often indicates manipulation or synthesis techniques. "
        explanation += f"The model detected {analysis['num_hot_regions']} highly suspicious regions with strong activation patterns. "
        explanation += "These patterns are characteristic of deepfake artifacts, face swapping, or digital manipulation."
        
    elif confidence > 0.85:
        explanation = f"{base_explanation}{focus_description}, suggesting potential digital manipulation. "
        explanation += f"The model identified {analysis['num_hot_regions']} suspicious regions that show inconsistencies "
        explanation += "typical of AI-generated or heavily edited content."
        
    else:
        explanation = f"{base_explanation}{focus_description}, indicating possible artificial generation. "
        explanation += f"While {analysis['num_hot_regions']} regions show suspicious patterns, "
        explanation += "the model has moderate confidence in this assessment."
    
    # Add distribution-specific details
    if analysis["distribution"] == "concentrated":
        explanation += " The suspicious patterns are highly localized, suggesting targeted manipulation of specific facial features."
    elif analysis["distribution"] == "dispersed":
        explanation += " The suspicious patterns are spread across the image, which may indicate comprehensive face synthesis or generation."
    
    return explanation

def generate_real_explanation(analysis: Dict, confidence: float) -> str:
    """Generate explanation for real image predictions."""
    uncertainty = 1 - confidence
    
    base_explanation = "The model believes this is a real image, but detected some uncertainty around the "
    
    # Determine uncertain areas
    uncertain_areas = []
    if analysis["center_focus"]:
        uncertain_areas.append("central facial features")
    if analysis["edge_focus"]:
        uncertain_areas.append("image edges and lighting")
    
    if not uncertain_areas:
        if analysis["distribution"] == "dispersed":
            uncertain_areas.append("various facial regions")
        else:
            uncertain_areas.append("specific areas of the face")
    
    uncertainty_description = " and ".join(uncertain_areas)
    
    # Build explanation based on confidence
    if uncertainty < 0.15:  # High confidence real (>85%)
        explanation = f"The model is highly confident this is a real image. Minor attention to {uncertainty_description} "
        explanation += f"represents normal variation in {analysis['num_hot_regions']} regions. "
        explanation += "These patterns are consistent with natural photography and authentic facial features."
        
    elif uncertainty < 0.3:  # Medium confidence real (70-85%)
        explanation = f"{base_explanation}{uncertainty_description}. "
        explanation += f"The model found {analysis['num_hot_regions']} regions that require closer examination, "
        explanation += "but overall patterns suggest authentic content with possible compression artifacts or lighting variations."
        
    else:  # Lower confidence real (<70%)
        explanation = f"{base_explanation}{uncertainty_description}. "
        explanation += f"The model identified {analysis['num_hot_regions']} regions with ambiguous patterns "
        explanation += "that could be due to image quality, lighting conditions, or minor post-processing effects."
    
    # Add distribution-specific details
    if analysis["distribution"] == "concentrated":
        explanation += " The uncertain patterns are localized, possibly due to lighting, shadows, or natural facial asymmetries."
    elif analysis["distribution"] == "dispersed":
        explanation += " The uncertain patterns are distributed across the image, which may be due to image compression or camera characteristics."
    
    return explanation

def generate_fake_recommendation(analysis: Dict, confidence: float) -> str:
    """Generate recommendation for fake image predictions."""
    if confidence > 0.95:
        recommendation = "This image appears to be AI-generated or digitally manipulated. "
        recommendation += "We strongly recommend treating this image as synthetic and not using it as evidence of real events or people. "
        recommendation += "Consider reverse image searching to find the original source."
        
    elif confidence > 0.85:
        recommendation = "This image shows strong signs of digital manipulation. "
        recommendation += "We recommend exercising caution and verifying the source before sharing or using this content. "
        recommendation += "Consider seeking additional verification through other sources."
        
    else:
        recommendation = "This image may be artificially generated or manipulated. "
        recommendation += "We recommend additional verification through multiple detection tools or expert analysis "
        recommendation += "before making any definitive conclusions about its authenticity."
    
    return recommendation

def generate_real_recommendation(analysis: Dict, confidence: float) -> str:
    """Generate recommendation for real image predictions."""
    uncertainty = 1 - confidence
    
    if uncertainty < 0.15:  # High confidence real
        recommendation = "This image appears to be authentic and unmanipulated. "
        recommendation += "The model shows high confidence in its authenticity, with only minor attention to natural variations. "
        recommendation += "However, always consider the source and context when evaluating image authenticity."
        
    elif uncertainty < 0.3:  # Medium confidence real
        recommendation = "This image likely represents authentic content, though some regions show minor inconsistencies. "
        recommendation += "These could be due to image compression, lighting conditions, or camera artifacts. "
        recommendation += "Consider the source credibility and cross-reference with other verification methods if needed."
        
    else:  # Lower confidence real
        recommendation = "While this image appears to be real, there are some ambiguous patterns that warrant caution. "
        recommendation += "We recommend additional verification, especially if the image is being used for important decisions. "
        recommendation += "Consider using multiple detection tools or seeking expert analysis for higher certainty."
    
    return recommendation