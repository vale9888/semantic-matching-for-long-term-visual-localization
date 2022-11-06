import os

from fine_grained_segmentation.utils.misc import get_global_opts
from fine_grained_segmentation.eval.cluster_images_in_folder import cluster_images_to_text



if __name__ == '__main__':
    global_opts = get_global_opts()

    # Set slice to run experiments (see Extended-CMU-Seasons README)
    sl = '6'

    # Add your repo folder
    my_repo_path = 'ADD HERE YOUR REPO PATH'

    image_types = ['database', 'query']
    for type in image_types:
        args = {
            'use_gpu': True,
            # 'miou' (miou over classes present in validation set), 'acc'
            'img_set': '',  # ox, cmu, cityscapes overwriter img_path, img_ext and save_folder_name. Set to empty string to ignore

            # THESE VALUES ARE ONLY USED IF 'img_set': ''
            'img_path': my_repo_path + '/data/Extended-CMU-Seasons/slice{}/{}'.format(sl, type),
            'img_ext': '.jpg',
            'save_folder_name': my_repo_path + '/data/Extended-CMU-Seasons/slice{}/semantic_masks/numeric/{}'.format(sl, type),
            'slice_path': my_repo_path + '/data/Extended-CMU-Seasons/slice{}'.format(sl),


            # specify this if using specific weight file
            'network_file': global_opts['network_file'] if 'network_file' in global_opts else '',

            'n_slices_per_pass': 10,
            'sliding_transform_step': 2/3.
        }

        network_folder = my_repo_path + '/Models/trained/fgsn/'
        query_names = os.listdir(my_repo_path + '/data/Extended-CMU-Seasons/slice{}/{}'.format(sl, type))

        network_file = network_folder +'cmu-100clusters.pth'
        # store image segmentations in .txt files
        cluster_images_to_text(network_file, args, sl, query_names, type)