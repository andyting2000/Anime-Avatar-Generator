import gradio as gr
import torch
from PIL import Image
import numpy as np
import cv2
from model import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator()
model.load_state_dict(torch.load(
    "weight.pt", map_location=device))
model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_image(image, use_face_detection=True):
    img = np.array(image)

    if use_face_detection:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            img = img[y1:y2, x1:x2]

    img = cv2.resize(img, (512, 512))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img = img / 127.5 - 1.0

    return img.to(device)


def postprocess_image(tensor):
    img = tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_anime_style(image, use_face_detection):
    try:
        if image is None:
            return None

        input_tensor = preprocess_image(image, use_face_detection)

        with torch.no_grad():
            output = model(input_tensor)

        result = postprocess_image(output)

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


custom_css = """
#main-container {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#header {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    text-align: center;
}

#header h1 {
    color: white;
    font-size: 32px;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

#header p {
    color: #e0f2fe;
    font-size: 16px;
    margin: 8px 0 0 0;
    opacity: 0.95;
}

.input-panel, .output-panel {
    background: rgba(51, 65, 85, 0.5) !important;
    border: 1px solid rgba(100, 116, 139, 0.4) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    backdrop-filter: blur(10px);
    height: 100%;
}

.gr-button-primary {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
}

.gr-form {
    background: rgba(51, 65, 85, 0.4) !important;
    border: 1px solid rgba(100, 116, 139, 0.4) !important;
    border-radius: 8px !important;
}

#info-box {
    background: rgba(37, 99, 235, 0.12) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

#info-box h3 {
    color: #60a5fa !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

#info-box li {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
}

.gr-checkbox {
    color: #e2e8f0 !important;
}

label {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

#footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 30px;
    padding: 20px;
    border-top: 1px solid rgba(100, 116, 139, 0.3);
}
"""

with gr.Blocks(elem_id="main-container") as demo:
    with gr.Row(elem_id="header"):
        with gr.Column():
            gr.Markdown("# Anime Avatar Generator")
            gr.Markdown(
                "Transform your selfie into a beautiful anime-style avatar.")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="input-panel"):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(
                type="pil",
                label="Upload your selfie",
                height=512
            )

        with gr.Column(scale=1, elem_classes="output-panel"):
            gr.Markdown("### Result")
            output_image = gr.Image(
                label="Anime-style avatar",
                height=512,
                type="pil",
                format="png"
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Options")
            face_detection = gr.Checkbox(
                label="Enable Face Detection (automatically crop and align face)",
                value=True,
                info="Recommended for better result"
            )

            generate_btn = gr.Button(
                "Generate Anime Style Avatar", variant="primary", size="lg")

    with gr.Row(elem_id="info-box"):
        gr.Markdown("""
        ### How to Use
        1. Upload a clear selfie (front-facing selfie works the best).
        2. Choose whether to enable face detection for automatic cropping.
        3. Click "Generate Anime Style" to transform your selfie into an anime-style avatar.
        4. Download your anime-style avatar using the download button in the output panel.
        
        ### About this Web App
        This web application uses AnimeGAN V2 to generate anime-style avatars. AnimeGAN V2 is a Generative Adversarial Network (GAN) to transform real photographs into anime-style artworks. The model learns the distinctive features of anime art through training on thousands of anime images, enabling it to apply stylistic elements like simplified shading, enhanced colors, and characteristic linework while preserving the original facial structure and identity of a person.
        """)

    with gr.Row(elem_id="footer"):
        gr.Markdown("**Developed by Andy Ting**")

    generate_btn.click(
        fn=generate_anime_style,
        inputs=[input_image, face_detection],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=gr.themes.Base())
