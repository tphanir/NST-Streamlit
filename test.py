import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np

# Load VGG19 with pretrained weights (CPU only)
from torchvision.models import vgg19, VGG19_Weights
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to("cpu")

for param in vgg.parameters():
    param.requires_grad_(False)

# Function to convert image to tensor
def imageToTensor(image):
    im_size = 400
    image = image.convert("RGB")  # Ensure RGB format
    in_transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
    ])
    image = in_transform(image).unsqueeze(0)
    return image

# Function to convert tensor back to image
def tensorToImage(tensor):
    image = tensor.clone().detach().cpu()
    image = image.squeeze()
    image = torch.permute(image, (1, 2, 0))
    image = image.numpy().clip(0, 1)
    return image

# Feature extraction
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in enumerate(model):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

# Gram matrix calculation
def gram_matrix(tensor):
    _, n_c, n_h, n_w = tensor.size()
    tensor = tensor.view(n_c, n_h * n_w)
    return torch.mm(tensor, tensor.t())

# Streamlit UI
st.title("🎨 Neural Style Transfer Web App")

content_file = st.file_uploader("📤 Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("📤 Upload Style Image", type=["jpg", "png"])

if st.button("🎨 Generate Styled Image") and content_file and style_file:
    content = imageToTensor(Image.open(content_file)).to("cpu")
    style = imageToTensor(Image.open(style_file)).to("cpu")

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Compute Gram matrices
    gm = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Style weights
    style_weights = {'conv1_1': 0.85, 'conv2_1': 0.56, 'conv3_1': 0.11,
                     'conv4_1': 0.15, 'conv5_1': 0.2}
    
    # Style transfer parameters
    alpha, beta = 100, 40

    # Generate styled image
    generated = content.clone().requires_grad_(True).to("cpu")
    optimizer = torch.optim.Adam([generated], lr=0.02)
    loss_fn = torch.nn.MSELoss()

    # Training loop (reduced for faster cloud execution)
    for _ in range(300):  # Reduce iterations for Streamlit Cloud
        target_features = get_features(generated, vgg)
        content_loss = loss_fn(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0
        for layer in style_weights:
            target_gram = gram_matrix(target_features[layer])
            style_gram = gm[layer]
            style_loss += style_weights[layer] * loss_fn(target_gram, style_gram)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    output_image = tensorToImage(generated)
    st.image(output_image, caption="🎉 Styled Image", use_column_width=True)
