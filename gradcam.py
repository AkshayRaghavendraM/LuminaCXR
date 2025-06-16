import torch
import numpy as np
import cv2
from torchvision import models
from PIL import Image

# ----------------------------- Device -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Classes -----------------------------
classes = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# ----------------------------- Custom DenseNet Forward -----------------------------
from torchvision.models.densenet import DenseNet

class CustomDenseNet(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features.clone())  # clone avoids inplace operation
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1)
        out = self.classifier(out)
        return out

# ----------------------------- Load Model -----------------------------
def load_model():
    model = CustomDenseNet(
        growth_rate=32, block_config=(6, 12, 24, 16),
        num_init_features=64, bn_size=4, drop_rate=0,
        num_classes=len(classes)
    )
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----------------------------- Grad-CAM Class -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        grads = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam) + 1e-8
        return cam
