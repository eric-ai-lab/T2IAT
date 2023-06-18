import numpy as np
import torch
import clip
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import gradio as gr

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def generate_image(text_prompt):
    images = pipe(text_prompt, num_images_per_prompt=10).images
    return images

def build_generation_block(prompt):
    with gr.Row(variant="compact"):
        text = gr.Textbox(
            label=prompt,
            show_label=False,
            max_lines=1,
            placeholder=prompt,
        ).style(
            container=False,
        )
        btn = gr.Button("Generate image").style(full_width=False)

    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    ).style(columns=[5], rows=[2], object_fit="contain", height="auto")
    
    btn.click(generate_image, text, gallery)

    return text, gallery

def compute_association_score(image_null, image_pos, image_neg):
    def compute_score(images):
        # print(images[0])
        features = [preprocess(Image.open(i['name'])) for i in images]
        features = torch.stack(features).to(device)
        with torch.no_grad():
            image_features = model.encode_image(features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    emb_null = compute_score(image_null)
    emb_pos  = compute_score(image_pos)
    emb_neg  = compute_score(image_neg)

    return np.mean(emb_pos @ emb_null.T) - np.mean(emb_neg @ emb_null.T)



with gr.Blocks() as demo:
    with gr.Group():
        gr.HTML("<h1 align='center'>T2IAT: Measuring Valence and Stereotypical Biases in Text-to-Image Generation")
        gr.HTML("<h1 align='center'><strong style='color:#A52A2A'>ACL 2023 (Findings)</strong></h1>")
        gr.HTML("<h2 align='center' style='color:#29A6A6'>Jialu Wang, Xinyue Gabby Liu, Zonglin Di, Yang Liu, Xin Eric Wang</h2>")
        gr.HTML("<h2 align='center'>University of California, Santa Cruz</h2>")
        gr.HTML("""
            <h2 align="center">
            <span style='display:inline'>
                <a href="https://arxiv.org/abs/2306.00905" class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                        <svg style="display:inline-block;font-size:inherit;height:1em;overflow:visible;vertical-align:-.125em" aria-hidden="true" focusable="false" data-prefix="fas" data-icon="file-pdf" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" data-fa-i2svg=""><path fill="currentColor" d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-29.1 24.5-34.9 40.2zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-8 171.8c-20-12.2-33.3-29-42.7-53.8 4.5-18.5 11.6-46.6 6.2-64.2-4.7-29.4-42.4-26.5-47.8-6.8-5 18.3-.4 44.1 8.1 77-11.6 27.6-28.7 64.6-40.8 85.8-.1 0-.1.1-.2.1-27.1 13.9-73.6 44.5-54.5 68 5.6 6.9 16 10 21.5 10 17.9 0 35.7-18 61.1-61.8 25.8-8.5 54.1-19.1 79-23.2 21.7 11.8 47.1 19.5 64 19.5 29.2 0 31.2-32 19.7-43.4-13.9-13.6-54.3-9.7-73.6-7.2zM377 105L279 7c-4.5-4.5-10.6-7-17-7h-6v128h128v-6.1c0-6.3-2.5-12.4-7-16.9zm-74.1 255.3c4.1-2.7-2.5-11.9-42.8-9 37.1 15.8 42.8 9 42.8 9z"></path></svg><!-- <i class="fas fa-file-pdf"></i> Font Awesome fontawesome.com -->
                    </span>
                    <span>Paper</span>
                </a>
            </span>
            </h2>
        """)

        gr.HTML("""
        <div>
        <p style="padding: 25px 200px; text-align: justify;">
            <strong>Abstract:</strong> In the last few years, text-to-image generative models have gained remarkable success in generating images with unprecedented quality accompanied by a breakthrough of inference speed. Despite their rapid progress, human biases that manifest in the training examples, particularly with regard to common stereotypical biases, like gender and skin tone, still have been found in these generative models. In this work, we seek to measure more complex human biases exist in the task of text-to-image generations. Inspired by the well-known Implicit Association Test (IAT) from social psychology, we propose a novel Text-to-Image Association Test (T2IAT) framework that quantifies the implicit stereotypes between concepts and valence, and those in the images. We replicate the previously documented bias tests on generative models, including morally neutral tests on flowers and insects as well as demographic stereotypical tests on diverse social attributes. The results of these experiments demonstrate the presence of complex stereotypical behaviors in image generations. 
        </p>
        </div>
        """)

        # gr.Image(
        #     "images/Text2ImgAssocationTest.png"
        # ).style(
        #     height=300,
        #     weight=400
        # )

    with gr.Group():
        gr.HTML("""
            <h3>First step: generate images with neutral prompts</h3>
        """)
        text_null, gallery_null = build_generation_block("Enter the neutral prompt.")
    
    with gr.Group():
        gr.HTML("""
            <h3>Second step: generate attribute-guided images by including the attributes into the prompts</h3> 
        """)
        text_pos, gallery_pos = build_generation_block("Enter your prompt with attribute A.")
        text_neg, gallery_neg = build_generation_block("Enter your prompt with attribute B.")

    with gr.Group():
        gr.HTML("<h3>Final step: compute the association score between your specified attributes!")

        with gr.Row():
            score = gr.Number(label="association score")
            btn = gr.Button("Compute Association Score!")
            btn.click(compute_association_score, [gallery_null, gallery_pos, gallery_neg], score)
        
        gr.HTML("<p>The absolute value of the association score represents the strength of the bias between the compared attributes, A and B, subject to the concepts that users choose in image generation. The higher score, the stronger the association, and vice versa.</p>")

if __name__ == "__main__":
    demo.queue(concurrency_count=3)
    demo.launch(title='T2IAT')

