import cv2
import torch
from torchvision import transforms
from PIL import Image
from argparse import Namespace
import sys
import os

sys.path.append("pixel2style2pixel")

from models.psp import pSp
from utils.common import tensor2im


def load_model():
    ckpt_path = "pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = ckpt_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    return net, opts


def capture_face_image(filename="face_input.jpg"):
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture your face.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture Your Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(filename, frame)
            print("Image captured.")
            break
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return transform(image).unsqueeze(0).cuda()


def generate_avatar(net, img_tensor, save_path="output_avatar.jpg"):
    with torch.no_grad():
        result = net(img_tensor, randomize_noise=False, return_latents=False)
        output_image = tensor2im(result[0])
        output_image.save(save_path)
        print(f"Stylized avatar saved as {save_path}")


if __name__ == "__main__":
    model, options = load_model()
    capture_face_image()
    input_tensor = preprocess_image("face_input.jpg")
    generate_avatar(model, input_tensor)
