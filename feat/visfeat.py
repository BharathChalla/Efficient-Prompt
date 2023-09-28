import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop, InterpolationMode

from clip import clip
from clip_models import load_openai_clip_model_processor, load_huggingface_clip_model_processor


def load_clip_text_and_image(backbone_name='clip-ViT-B-16'):
    if backbone_name == 'openai/clip-vit-base-patch16':
        model, processor = load_openai_clip_model_processor(backbone_name=backbone_name)
        return model, processor
    elif backbone_name == 'clip-ViT-B-16':
        # Ref: https://huggingface.co/sentence-transformers/clip-ViT-B-16
        # Load CLIP model
        model = load_huggingface_clip_model_processor(backbone_name=backbone_name)
        return model
    raise NotImplementedError('Only supports openAI/clip-ViT-B-16 and HuggingFace clip-ViT-B-16 now!')


def load_clip_cpu(model_name=None):
    if model_name is None:
        model_name = clip.available_models()[-1]  # ViT-B/16
    model_url = clip._MODELS[model_name]

    model_file_name = model_url.split('/')[-1]
    # Download the model manually from the link below and put it in the models folder
    models_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models'))
    model_path = os.path.join(models_dir, model_file_name)

    if not os.path.exists(model_path):
        print('Downloading the model {} from {} ...'.format(model_name, model_url))
        os.makedirs(models_dir, exist_ok=True)
        clip._download(model_url, models_dir)
        print('Downloading the model {} is done!'.format(model_name))
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model = clip.build_model(state_dict or model.state_dict())
    return model


def load_clip_model(model_name):
    print("Loading CLIP Model Name:", model_name)
    model = clip.load(model_name, jit=False)[0].eval()

    models_dir = os.path.expanduser("~/.cache/clip")
    print("Model download directory:", models_dir)

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


def get_videos(vid_name, read_path, cen_trans):
    all_frames = []
    videoins = os.path.join(read_path, vid_name)
    vvv = cv2.VideoCapture(videoins)
    if not vvv.isOpened():
        print('Video is not opened! {}'.format(videoins))
    else:
        fps = vvv.get(cv2.CAP_PROP_FPS)
        total_frame_number = vvv.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_size = (int(vvv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vvv.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        duration = total_frame_number // fps
        print(
            'video %s - fps: %.2f, total_frame_number: %d, frame_size: %s, duration: %d' %
            (vid_name, fps, total_frame_number, frame_size, duration)
        )
        # ToDo: uncomment the following line to Debug for Multi Model Instances on 1 GPU
        # total_frame_number = 2000
        if total_frame_number != 0:
            for _ in range(int(total_frame_number)):
                rval, frame = vvv.read()
                if frame is not None:
                    img = Image.fromarray(frame.astype('uint8')).convert('RGB')
                    img_trans = cen_trans(img).numpy()
                    all_frames.append(img_trans)
                if len(all_frames) % 1000 == 0:
                    print('video %s - transformed %d frames!' % (vid_name, len(all_frames)))

    return np.array(all_frames)


def extract_video_clip_features():
    max_len = 1000  # the maximum number of video frames that GPU can process
    dataset_dir = '/data/error_dataset'
    features_dir = os.path.join(dataset_dir, 'features')
    save_path = os.path.join(features_dir, 'CLIP')
    os.makedirs(save_path, exist_ok=True)

    data_path = os.path.join(dataset_dir, 'videos')
    os.chdir(data_path)
    all_videos = os.listdir()
    all_videos.sort()
    cen_trans = transform_center()

    num_of_videos = len(all_videos)
    half_num_of_videos = num_of_videos // 2
    start_idx = 0

    # load CLIP pre-trained parameters
    device = 'cuda'
    # model = load_clip_text_and_image()
    clip_model = load_clip_cpu('ViT-B/16')
    clip_model.to(device)
    # clip_model.eval()
    for clip_param in clip_model.parameters():
        clip_param.requires_grad = False

    for vid in range(start_idx, len(all_videos)):
        features_path = os.path.join(save_path, all_videos[vid][:-4] + '.npy')
        video_name = all_videos[vid]
        if os.path.exists(features_path):
            print('video %d - %s has been done!' % (vid, video_name))
            continue
        print('transform video %d - %s video has been started!' % (vid, video_name))
        video_load_start_time = time.time()
        vidone = get_videos(all_videos[vid], data_path, cen_trans)  # shape = (T,3,224,224)
        video_load_end_time = time.time()
        print('transform video %d - %s video has been done!' % (vid, video_name))
        print('video %s frames transform time: %.2fs' % (all_videos[vid], video_load_end_time - video_load_start_time))

        video_feat_start_time = time.time()
        vidinsfeat = []
        for k in range(int(len(vidone) / max_len) + 1):  # if the video is too long, split the video
            segframes = torch.from_numpy(vidone[k * max_len:(k + 1) * max_len]).to(device)
            vis_feats = clip_model.encode_image(segframes)
            vidinsfeat = vidinsfeat + vis_feats.cpu().numpy().tolist()
        vidinsfeat = np.array(vidinsfeat)  # shape = (T,512)

        assert (len(vidinsfeat) == len(vidone))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, all_videos[vid][:-4] + '.npy'), vidinsfeat)

        print('visual features of %d video have been done!' % vid)
        video_feat_end_time = time.time()
        print('video %s feature extract time: %.2fs' % (all_videos[vid], video_feat_end_time - video_feat_start_time))

    print('all %d visual features have been done!' % len(all_videos))
    pass


if __name__ == "__main__":
    extract_video_clip_features()
