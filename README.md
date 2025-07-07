# Face2Game-GAN: AI-Powered Game Character Generator using GAN-based Facial Stylization

Face2Game-GAN is a deep learning pipeline that captures a user's face through a webcam or uploaded image and generates a personalized, stylized game avatar using a pretrained StyleGAN2 model. It leverages GAN inversion techniques through the pSp encoder to map real images into the latent space of the generator.

## Features

- Capture face image using webcam
- Encode real face images into latent space using pSp (pixel2style2pixel)
- Generate high-quality stylized game avatars using StyleGAN2
- Save the output character image locally
- Built with modularity for future integration with game engines or web applications

## Project Structure

face2game_gan/
├── generate.py # Main execution script
├── face_input.jpg # Captured face image (auto-generated)
├── output_avatar.jpg # Stylized avatar output
└── pixel2style2pixel/ # pSp encoder repository
├── pretrained_models/
│ └── psp_ffhq_encode.pt # Pretrained pSp+StyleGAN2 model checkpoint

## Getting Started

### 1. Clone the Required Repository

Clone the official pSp (pixel2style2pixel) repository and install dependencies:

```bash
git clone https://github.com/eladrich/pixel2style2pixel.git
cd pixel2style2pixel
pip install -r requirements.txt
Download the pretrained psp_ffhq_encode.pt checkpoint from:

https://github.com/eladrich/pixel2style2pixel#pre-trained-models
