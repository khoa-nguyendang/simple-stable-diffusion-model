import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

def inference(prompt: str,
              negative_prompt: str,
              steps: int = 20,
              width: int = 512,
              height: int = 512,
              w_path: str = '') -> Image:
    finetuned_model = keras_cv.models.StableDiffusion(img_width=width, img_height=height)
    finetuned_model.diffusion_model.load_weights(w_path)
    # prompts = ["Xe hơi màu đỏ", "Xe hơi và cô gái", "Xe đua"]
    seed = np.random()
    return finetuned_model.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            batch_size=1,
            unconditional_guidance_scale=40
        )


