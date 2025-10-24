"""
AI-Based Counterfeit Bike Parts Detection System
Complete version with Examples section, fixed table, and model loading
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

POSSIBLE_MODEL_PATHS = [
    'best.pt',
    'weights/best.pt',
    'last.pt',
    'weights/last.pt',
    'bike_parts_classifier/yolov8m_counterfeit_detection/weights/best.pt',
]

def load_model():
    """Load trained YOLOv8 classification model"""
    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print(f"✅ Model loaded from: {model_path}")
                return model, model_path
            except Exception as e:
                print(f"⚠️  Failed to load {model_path}: {e}")
    return None, None

model, loaded_model_path = load_model()

# Class information
CLASS_INFO = {
    'air_filter_genuine': {'part_type': 'Air Filter', 'status': 'Genuine', 'color': '#10b981', 'icon': '✅'},
    'air_filter_fake': {'part_type': 'Air Filter', 'status': 'Counterfeit', 'color': '#ef4444', 'icon': '⚠️'},
    'helmet_genuine': {'part_type': 'Helmet', 'status': 'Genuine', 'color': '#10b981', 'icon': '✅'},
    'helmet_fake': {'part_type': 'Helmet', 'status': 'Counterfeit', 'color': '#ef4444', 'icon': '⚠️'},
    'spark_plug_genuine': {'part_type': 'Spark Plug', 'status': 'Genuine', 'color': '#10b981', 'icon': '✅'},
    'spark_plug_fake': {'part_type': 'Spark Plug', 'status': 'Counterfeit', 'color': '#ef4444', 'icon': '⚠️'}
}

# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def classify_bike_part(image, show_top_predictions=3):
    """Classify bike part with fixed table display"""
    
    if model is None:
        error = "❌ Model not found. Place best.pt or last.pt in current directory!"
        return None, error, "<p style='color: red;'>Model not loaded</p>", None
    
    if image is None:
        return None, "Upload an image to start", "", None
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    try:
        # Predict
        results = model.predict(source=image, verbose=False)
        result = results[0]
        probs = result.probs
        
        # Top prediction
        top1_idx = probs.top1
        top1_conf = float(probs.top1conf)
        class_name = result.names[top1_idx]
        class_info = CLASS_INFO.get(class_name, {
            'part_type': 'Unknown', 'status': 'Unknown', 'color': '#6b7280', 'icon': '❓'
        })
        
        # === CREATE RESULT IMAGE ===
        result_image = image_np.copy()
        h, w = result_image.shape[:2]
        
        # Overlay
        overlay = result_image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        result_image = cv2.addWeighted(result_image, 0.6, overlay, 0.4, 0)
        
        # Text overlay
        cv2.putText(result_image, f"{class_info['icon']} {class_info['status'].upper()}", 
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(result_image, class_info['part_type'], 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_image, f"Confidence: {top1_conf*100:.1f}%", 
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Border
        border_color = tuple(int(class_info['color'].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(result_image, (0, 0), (w-1, h-1), border_color, 10)
        
        # === SUMMARY ===
        summary = f"{class_info['icon']} **Classification Result**\n\n"
        summary += f"**Part Type:** {class_info['part_type']}\n"
        summary += f"**Status:** {class_info['status']}\n"
        summary += f"**Confidence:** {top1_conf*100:.1f}%\n\n"
        summary += "⚠️ **Do not use**" if class_info['status'] == 'Counterfeit' else "✅ **Appears genuine**"
        
        # === FIXED TABLE WITH VISIBLE TEXT ===
        details_html = f"""
        <div style='border: 3px solid {class_info['color']}; border-radius: 10px; padding: 20px; background: white;'>
            <h3 style='color: {class_info['color']};'>{class_info['icon']} Primary Classification</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background: {class_info['color']}20;'>
                    <td style='padding: 15px; font-weight: bold; color: #1f2937;'>Part Type</td>
                    <td style='padding: 15px; color: #1f2937;'>{class_info['part_type']}</td>
                </tr>
                <tr>
                    <td style='padding: 15px; font-weight: bold; color: #1f2937;'>Status</td>
                    <td style='padding: 15px; color: {class_info['color']}; font-weight: bold;'>{class_info['status']}</td>
                </tr>
                <tr style='background: {class_info['color']}20;'>
                    <td style='padding: 15px; font-weight: bold; color: #1f2937;'>Confidence</td>
                    <td style='padding: 15px; color: #1f2937;'>{top1_conf*100:.2f}%</td>
                </tr>
            </table>
        </div>
        
        <h3 style='color: #1f2937; margin-top: 20px;'>📊 Top {show_top_predictions} Predictions</h3>
        <table style='width: 100%; border-collapse: collapse; background: white;'>
            <tr style='background: #4a5568; color: white;'>
                <th style='padding: 10px; border: 1px solid #cbd5e0;'>Rank</th>
                <th style='padding: 10px; border: 1px solid #cbd5e0;'>Part Type</th>
                <th style='padding: 10px; border: 1px solid #cbd5e0;'>Status</th>
                <th style='padding: 10px; border: 1px solid #cbd5e0;'>Confidence</th>
                <th style='padding: 10px; border: 1px solid #cbd5e0;'>Bar</th>
            </tr>
        """
        
        # Top predictions
        top_indices = probs.top5[:show_top_predictions]
        for rank, idx in enumerate(top_indices, 1):
            cls_name = result.names[idx]
            cls_conf = float(probs.data[idx])
            cls_info = CLASS_INFO.get(cls_name, CLASS_INFO['spark_plug_genuine'])
            
            bg = "#f9fafb" if rank % 2 == 0 else "white"
            details_html += f"""
            <tr style='background: {bg};'>
                <td style='padding: 10px; border: 1px solid #e5e7eb; color: #1f2937; font-weight: bold;'>{rank}</td>
                <td style='padding: 10px; border: 1px solid #e5e7eb; color: #1f2937;'>{cls_info['part_type']}</td>
                <td style='padding: 10px; border: 1px solid #e5e7eb; color: {cls_info['color']}; font-weight: bold;'>{cls_info['status']}</td>
                <td style='padding: 10px; border: 1px solid #e5e7eb; color: #1f2937; font-weight: bold;'>{cls_conf*100:.2f}%</td>
                <td style='padding: 10px; border: 1px solid #e5e7eb;'>
                    <div style='background: #e5e7eb; border-radius: 4px; width: 200px;'>
                        <div style='width: {cls_conf*100}%; background: {cls_info['color']}; height: 25px; display: flex; align-items: center; justify-content: center;'>
                            <span style='color: white; font-size: 11px; font-weight: bold;'>{cls_conf*100:.1f}%</span>
                        </div>
                    </div>
                </td>
            </tr>
            """
        details_html += "</table>"
        
        # === CHART ===
        top_names = [result.names[idx] for idx in top_indices]
        top_confs = [float(probs.data[idx]) * 100 for idx in top_indices]
        top_colors = [CLASS_INFO.get(n, {}).get('color', '#6b7280') for n in top_names]
        chart_labels = [f"{CLASS_INFO.get(n, {}).get('part_type', 'Unknown')} - {CLASS_INFO.get(n, {}).get('status', 'Unknown')}" 
                        for n in top_names]
        
        fig = go.Figure(data=[
            go.Bar(
                y=chart_labels[::-1],
                x=top_confs[::-1],
                orientation='h',
                marker=dict(color=top_colors[::-1]),
                text=[f"{c:.1f}%" for c in top_confs[::-1]],
                textposition='inside',
                textfont=dict(color='white', size=14)
            )
        ])
        
        fig.update_layout(
            title='Confidence Levels',
            xaxis=dict(title='Confidence (%)', range=[0, 100]),
            yaxis=dict(title=''),
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return result_image, summary, details_html, fig
        
    except Exception as e:
        return None, f"❌ Error: {e}", f"<p style='color: red;'>{e}</p>", None

# ============================================================================
# EXAMPLE IMAGES FUNCTION
# ============================================================================

def get_example_images():
    """Dynamically find example images from dataset"""
    
    # Try multiple possible locations
    possible_dirs = [
        'dataset_clean/valid',
        'dataset_clean/test',
        'dataset1'
    ]
    
    examples = []
    
    for base_dir in possible_dirs:
        if not os.path.exists(base_dir):
            continue
        
        # Look for example images in subdirectories
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    examples.append([full_path, 3])  # [image_path, top_k]
                    
                    # Limit to 6 examples
                    if len(examples) >= 6:
                        return examples
    
    # If no dataset found, return None (Examples section won't show)
    return examples if examples else None

# ============================================================================
# GRADIO INTERFACE WITH EXAMPLES
# ============================================================================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    title="Counterfeit Detection",
    css=".gradio-container {font-family: Arial;}"
) as demo:
    
    gr.Markdown("# 🔍 AI-Based Counterfeit Bike Parts Detection\n### Upload Spark Plug, Helmet, or Air Filter image")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            input_image = gr.Image(label="Drop image or Click", type="pil", height=450)
            
            with gr.Accordion("⚙️ Settings", open=False):
                top_k = gr.Slider(1, 6, value=3, step=1, label="Top Predictions")
            
            with gr.Row():
                classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Result")
            output_image = gr.Image(label="Analysis", type="numpy", height=450)
            summary_text = gr.Markdown(value="Upload image to start...")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Detailed Analysis")
            results_html = gr.HTML(value="<p style='text-align: center; color: #6b7280; padding: 40px;'>Awaiting classification...</p>")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Confidence Chart")
            confidence_chart = gr.Plot()
    
    # ========== EXAMPLES SECTION (RESTORED) ==========
    example_images = get_example_images()
    
    if example_images:
        gr.Markdown("---")
        gr.Markdown("### 🖼️ Example Images")
        gr.Markdown("Click any example below to test the system")
        
        gr.Examples(
            examples=example_images,
            inputs=[input_image, top_k],
            outputs=[output_image, summary_text, results_html, confidence_chart],
            fn=classify_bike_part,
            cache_examples=False,
            label="Click to Load Example"
        )
    
    # Button actions
    classify_btn.click(
        fn=classify_bike_part,
        inputs=[input_image, top_k],
        outputs=[output_image, summary_text, results_html, confidence_chart]
    )
    
    clear_btn.click(
        fn=lambda: [None, None, "Upload image to start...", 
                    "<p style='text-align: center; color: #6b7280; padding: 40px;'>Awaiting classification...</p>", None],
        outputs=[input_image, output_image, summary_text, results_html, confidence_chart]
    )
    
    # Footer
    gr.Markdown("""
        ---
        ### 📚 Project Information
        **Team Members:** Rubeshwaran (1118), Saravanan (1127), Veerachandru (1148)  
        **Coordinators:** Mrs. Maheswari A, Ms. Swathi  
        **Technology:** YOLOv8 Classification | **Accuracy:** 96-98%  
        **Supported Parts:** Spark Plugs, Helmets, Air Filters
        
        ⚠️ **Disclaimer:** Educational project. Consult professionals for purchase decisions.
    """)

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COUNTERFEIT BIKE PARTS DETECTION SYSTEM")
    print("="*80)
    
    if model:
        print(f"✅ Model: {loaded_model_path}")
        print(f"✅ Classes: {list(model.names.values())}")
    else:
        print("⚠️  Model not found!")
        print("   Place best.pt or last.pt in current directory")
    
    # Check for examples
    examples_found = get_example_images()
    if examples_found:
        print(f"✅ Found {len(examples_found)} example images")
    else:
        print("⚠️  No example images found (Examples section hidden)")
    
    print("="*80 + "\n")
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=False
    )

