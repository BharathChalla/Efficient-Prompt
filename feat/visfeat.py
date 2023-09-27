import os
import time

import cv2
import torch
import numpy as np
from PIL import Image
from clip import clip

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop, InterpolationMode


def load_clip_text_and_image(backbone_name='clip-ViT-B-16'):
    if backbone_name == 'openai/clip-vit-base-patch16':
        # REf: https://huggingface.co/openai/clip-vit-base-patch16
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        image = Image.open("path_to_image")
        text_inputs = ["a photo of a cat", "a photo of a dog"]
        inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # Ref: https://huggingface.co/sentence-transformers/clip-ViT-B-16
    if backbone_name != 'clip-ViT-B-16':
        raise NotImplementedError('Only support clip-ViT-B-16 now!')
    # Load CLIP model
    clip_model_vit_b16 = SentenceTransformer('clip-ViT-B-16')

    # example:
    """
    # Encode an image:
    img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))

    # Encode text descriptions
    text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

    # Compute cosine similarities 
    cos_scores = util.cos_sim(img_emb, text_emb)
    print(cos_scores)
    """

    return clip_model_vit_b16


def load_clip_cpu(backbone_name):
    if backbone_name != 'ViT-B-16':
        raise NotImplementedError('Only support ViT-B-16 now!')

    # models_dir = os.path.expanduser("~/.cache/torch/clip")
    models_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models'))
    model_path = os.path.join(models_dir, 'ViT-B-16.pt')
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model = clip.build_model(state_dict or model.state_dict())

    return model


def transform_center():
    # interp_mode = Image.BICUBIC
    interp_mode = InterpolationMode.BICUBIC
    tfm_test = []
    tfm_test += [Resize(224, interpolation=interp_mode)]
    tfm_test += [CenterCrop((224, 224))]
    tfm_test += [ToTensor()]
    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    tfm_test += [normalize]
    tfm_test = Compose(tfm_test)

    return tfm_test


def get_videos(vidname, read_path):
    allframes = []
    videoins = os.path.join(read_path, vidname)
    vvv = cv2.VideoCapture(videoins)
    if not vvv.isOpened():
        print('Video is not opened! {}'.format(videoins))
    else:
        fps = vvv.get(cv2.CAP_PROP_FPS)
        totalFrameNumber = vvv.get(cv2.CAP_PROP_FRAME_COUNT)
        size = (int(vvv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vvv.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        second = totalFrameNumber//fps

        if totalFrameNumber != 0:
            for _ in range(int(totalFrameNumber)):
                rval, frame = vvv.read()
                if frame is not None:
                    img = Image.fromarray(frame.astype('uint8')).convert('RGB')
                    imgtrans = centrans(img).numpy()
                    allframes.append(imgtrans)

    return np.array(allframes)


if __name__ == "__main__":
    maxlen = 2000                                           # the maximum number of video frames that GPU can process
    dataset_dir = '/data/error_dataset'
    features_dir = os.path.join(dataset_dir, 'features')
    save_path = os.path.join(features_dir, 'CLIP')
    os.makedirs(save_path, exist_ok=True)

    datapath = os.path.join(dataset_dir, 'videos')
    os.chdir(datapath)
    allvideos = os.listdir()
    allvideos.sort()
    centrans = transform_center()

    # load CLIP pre-trained parameters
    device = 'cuda'
    model = load_clip_text_and_image()
    clip_model = load_clip_cpu('ViT-B-16')
    clip_model.to(device)
    for paramclip in clip_model.parameters():
        paramclip.requires_grad = False

    for vid in range(len(allvideos)):
        features_path = os.path.join(save_path, allvideos[vid][:-4] + '.npy')
        video_name = allvideos[vid]
        if os.path.exists(features_path):
            print('video %d - %s has been done!' % (vid, video_name))
            continue
        print('transform video %d - %s video has been started!' % (vid, video_name))
        video_load_start_time = time.time()
        vidone = get_videos(allvideos[vid], datapath)      # shape = (T,3,224,224)
        video_load_end_time = time.time()
        print('transform video %d - %s video has been done!' % (vid, video_name))
        print('video %s frames transform time: %.2fs' % (allvideos[vid], video_load_end_time - video_load_start_time))

        video_feat_start_time = time.time()
        vidinsfeat = []
        for k in range(int(len(vidone)/maxlen)+1):         # if the video is too long, split the video
            segframes = torch.from_numpy(vidone[k*maxlen:(k+1)*maxlen]).to(device)
            vis_feats = clip_model.encode_image(segframes)
            vidinsfeat = vidinsfeat + vis_feats.cpu().numpy().tolist()
        vidinsfeat = np.array(vidinsfeat)                  # shape = (T,512)

        assert(len(vidinsfeat) == len(vidone))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, allvideos[vid][:-4] + '.npy'), vidinsfeat)

        print('visual features of %d video have been done!' % vid)
        video_feat_end_time = time.time()
        print('video %s feature extract time: %.2fs' % (allvideos[vid], video_feat_end_time - video_feat_start_time))

    print('all %d visual features have been done!' % len(allvideos))
