

import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf


def inference(w_path: str, is_url: bool) -> dict:
    weights_path = w_path
    if is_url:
        weights_path = tf.keras.utils.get_file(origin=w_path)
    img_height = img_width = 512
    finetuned_model = keras_cv.models.StableDiffusion(
        img_width=img_width, img_height=img_height
    )
    # We just reload the weights of the fine-tuned diffusion model.
    finetuned_model.diffusion_model.load_weights(weights_path)

    prompts = ["Xe hơi màu đỏ", "Xe hơi và cô gái", "Xe đua"]
    images_to_generate = 3
    outputs = {}

    for prompt in prompts:
        generated_images = finetuned_model.text_to_image(
            prompt,
            batch_size=images_to_generate, 
            unconditional_guidance_scale=40
        )
        outputs.update({prompt: generated_images})
    
    return outputs

def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(title, fontsize=12)
        plt.axis("off")

def test():
    #url = "https://drive.google.com/file/d/1AOzAjrj6FgOaaxgTR1RUvwDjwjNXZ-jK/view?usp=drive_link"
    w_path = "./finetuned_stable_diffusion.h5" #input valid path of model
    outputs = inference(w_path, False)
    for prompt in outputs:
        plot_images(outputs[prompt], prompt)