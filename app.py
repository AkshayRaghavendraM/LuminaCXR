import os
import torch
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import cv2

from gradcam import GradCAM, model, classes
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected image", 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    # Grad-CAM
    gradcam = GradCAM(model, model.features.denseblock4)
    cam = gradcam.generate(input_tensor, class_idx=pred_idx)

    # Prepare overlay
    image_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    cam_path = os.path.join(UPLOAD_FOLDER, "cam_" + file.filename)
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return render_template(
        'result.html',
        prediction=classes[pred_idx],
        confidence=f"{confidence * 100:.2f}%",
        uploaded_image=file.filename,
        cam_image="cam_" + file.filename
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
