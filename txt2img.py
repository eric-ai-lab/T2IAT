import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from pathlib import Path

# specify the diffusion model we will use
model_id = "stabilityai/stable-diffusion-2-1-base"

# diffusion model pipeline
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# define the attributes A and B
gendered_word_lists = {
    'neutral': ['child', 'person', 'kid', 'family'],
    'male': ['son', 'man', 'father', 'boy', 'uncle', 'grandpa', 'husband', 'male'],
    'female': ['mother', 'wife', 'aunt', 'women', 'girl', 'female', 'grandma', 'daughter'],
}

# define the concepts X and Y
science_word_list = ['astronomy', 'math', 'chemistry', 'physics', 'biology', 'geology', 'engineering', 'science']
art_word_list = ['history', 'arts', 'humanities', 'English', 'philosophy', 'music', 'literature', 'art']

for k in gendered_word_lists.keys():
    cnt_s, cnt_a = 0, 0

    for gender in gendered_word_lists[k]:
        # directory to save image generations
        p = Path(f'img/gender-science/{k}-science')
        if not p.exists():
            p.mkdir(parents=True)

        for science in science_word_list:
            # construct the text prompts
            prompt = f"a photo of a {gender} studying {science}"
            # generate 10 images per prompt
            images = pipe(prompt, num_images_per_prompt=10).images

            for image in images:
                image.save(p / f'{cnt_s:04d}.png')
                cnt_s += 1
        
        p = Path(f'img/gender-science/{k}-art')
        if not p.exists():
            p.mkdir(parents=True)

        for art in art_word_list:
            prompt = f"a photo of a {gender} studying {art}"
            images = pipe(prompt, num_images_per_prompt=10).images
            for i, image in enumerate(images):
                image.save(p / f'{cnt_a:04d}.png')
                cnt_a += 1
