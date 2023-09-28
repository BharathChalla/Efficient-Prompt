import os


def rename_features():
    features_path = "/data/error_dataset/features/CLIP"
    os.chdir(features_path)
    all_feature_files = os.listdir()
    for feat_file in all_feature_files:
        if feat_file.endswith('.npy'):
            if '_360p' in feat_file:
                continue
            feat = feat_file.split('.')[0]
            new_feat = feat + '_360p.npy'
            os.rename(feat_file, new_feat)
            print('rename %s to %s' % (feat, new_feat))
    pass


if __name__ == '__main__':
    rename_features()
