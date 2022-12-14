# Script originally from https://github.com/maunzzz/fine-grained-segmentation-networks,
# licensed as in the LICENSE file of the above repository (Attribution-NonCommercial 4.0 International).

from utils.save_feature_positions import save_feature_positions
from utils.write_reference_im_list import write_reference_im_list
from utils.write_lists_for_corr import write_lists_for_corr

corr_set = 'cmu'

write_lists_for_corr(corr_set)
write_reference_im_list(corr_set)
save_feature_positions(corr_set)
