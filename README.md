# README

This is the code implementation for Text-to-Image Association Test (T2IAT). ![An example bias test instantiated on Gender-Science. The text prompt ``A photo of a child studying astronomy'' is constructed to generate neutral images. Then the gender-neutral word ``child'' is replaced with gendered words to generate attribute-specific images. We calculate the average difference in the distance between the neutral and attribute-specific images as a measure of association.](images/Text2ImgAssocationTest.png)

To run the script for image generations,

```
python3 txt2img.py
```

To run the association test with the image generations, go through the `bias-test.ipynb` jupyter notebook.

To deploy the gradio demo, run

```
gradio demo.py
```

and open `localhost:7860` in your local web browser.

## Requirements


```
pip3 install git+https://github.com/openai/CLIP.git
pip3 install --upgrade diffusers[torch]
pip3 install gradio  # only used for demo
```


## Citation

```
@inproceedings{wang-etal-2022-assessing,
    title = "T2IAT: Measuring Valence and Stereotypical Biases in Text-to-Image Generation",
    author = "Wang, Jialu  and
      Liu, Xinyue Gabby  and
      Di, Zonglin  and
      Liu, Yang  and
      Wang, Xin",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = July,
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```

