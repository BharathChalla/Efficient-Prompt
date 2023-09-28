import requests

from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel


def load_openai_clip_model_processor(
        backbone_name='openai/clip-vit-base-patch16',
        test_model=False, image=None, text_inputs=None
):
    # Ref: https://huggingface.co/openai/clip-vit-base-patch16
    if backbone_name != 'openai/clip-vit-base-patch16':
        raise NotImplementedError('Only support openai/clip-vit-base-patch16 now!')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Test the model
    if test_model:
        if image is None:
            url = "https://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
        if text_inputs is None:
            text_inputs = ["a photo of a cat", "a photo of a dog"]

        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        print("Label probs:", probs)

    return model, processor


def load_huggingface_clip_model_processor(
        backbone_name='clip-ViT-B-16',
        test_model=False, image=None, text_inputs=None
):
    if backbone_name != 'clip-ViT-B-16':
        raise NotImplementedError('Only support clip-ViT-B-16 now!')
    # Ref: https://huggingface.co/sentence-transformers/clip-ViT-B-16
    # Load CLIP model
    model = SentenceTransformer(backbone_name)

    # example:
    if test_model:
        if image is None:
            url = ("https://raw.githubusercontent.com/UKPLab/sentence-transformers/"
                   "master/examples/applications/image-search/two_dogs_in_snow.jpg")
            image = Image.open(requests.get(url, stream=True).raw)
        if text_inputs is None:
            text_inputs = ['Two dogs in the snow', 'A cat on a table', 'A picture of London at night']

        # Encode an image:
        img_emb = model.encode(image, convert_to_tensor=True)

        # Encode text descriptions
        text_emb = model.encode(text_inputs)

        # Compute cosine similarities
        cos_scores = util.cos_sim(img_emb, text_emb)
        print(cos_scores)

    return model
