import torch
from torch import nn
import numpy as np
from fine_grained_segmentation.clustering.clustering import preprocess_features
from scipy.spatial.distance import cdist


class ClusterCorrespondenceLoss(nn.Module):
    def __init__(self, input_size, size_average=True):
        super(ClusterCorrespondenceLoss, self).__init__()

        self.size_average = size_average ## True
        self.input_size = input_size ## 713x713
        self.loss = nn.CrossEntropyLoss(
            reduction='elementwise_mean', ignore_index=255)

    def forward(self, inputs_ref, inputs_other, inputs_feat_ref, inds_ref, inds_other, cluster_info=None, cluster_labels=None):
        ## To compute the loss you are given the two images predictions + input_feat_ref is None for us + point correspondences +
        ## + cluster labels for the points
        ## Tensor.view is basically reshape. This apparently flattens the dimensions greater than 2.
        ## I might be wrong, but I think the first dimension is the batch size. Thus all the image is then flattened along the HxW dims?
        inp_ref_flat = inputs_ref.view(
            [inputs_ref.size(0), inputs_ref.size(1), -1])
        inp_other_flat = inputs_other.view(
            [inputs_ref.size(0), inputs_other.size(1), -1])

        ## input_feat_ref are the features from the ref image I think, provided to go fetch the 2D-2D labels?
        if cluster_labels is None:
            inputs_feat_ref_flat = inputs_feat_ref.view(
                [inputs_feat_ref.size(0), inputs_feat_ref.size(1), -1])
            cluster_labels = []

        loss = 0.0
        ## For every image in the batch?
        for b in range(inputs_ref.size(0)):
            ## the points of this specific instance
            b_inds_ref = inds_ref[b]
            b_inds_other = inds_other[b]

            # rescale correspondence indices to feature size ## basically we recover the HxW dims
            if self.input_size != [inputs_ref.size(2), inputs_ref.size(3)]:
                b_inds_ref = b_inds_ref.type(torch.float32)
                b_inds_other = b_inds_other.type(torch.float32)

                ## might need to reshape  to make sure all dims are consistent. I am not sure if it is performing interpolation that was mentioned in the paper
                b_inds_ref[0, :] = b_inds_ref[0, :] / \
                    self.input_size[1]*inputs_ref.size(3)
                b_inds_ref[1, :] = b_inds_ref[1, :] / \
                    self.input_size[0]*inputs_ref.size(2)
                b_inds_other[0, :] = b_inds_other[0, :] / \
                    self.input_size[1]*inputs_ref.size(3)
                b_inds_other[1, :] = b_inds_other[1, :] / \
                    self.input_size[0]*inputs_ref.size(2)

            b_inds_ref = (b_inds_ref + .5).type(torch.int64)
            b_inds_other = (b_inds_other + .5).type(torch.int64)

            lin_inds_ref = inputs_ref.size(
                2)*b_inds_ref[1, :] + b_inds_ref[0, :]
            lin_inds_other = inputs_ref.size(
                2)*b_inds_other[1, :] + b_inds_other[0, :]

            valid_inds = (b_inds_ref[0, :] < inputs_ref.size(3)) & (b_inds_ref[0, :] >= 0) & (b_inds_ref[1, :] < inputs_ref.size(2)) & (b_inds_ref[1, :] >= 0) & (
                b_inds_other[0, :] < inputs_ref.size(3)) & (b_inds_other[0, :] >= 0) & (b_inds_other[1, :] < inputs_ref.size(2)) & (b_inds_other[1, :] >= 0)

            lin_inds_ref = lin_inds_ref[valid_inds]
            lin_inds_other = lin_inds_other[valid_inds]

            # Get cluster classes for features
            ## This is not used in our case though
            if cluster_info is not None:
                # This could probably be done a lot nicer
                numpy_features = np.transpose(
                    inputs_feat_ref_flat[b, :, lin_inds_ref].clone().detach().cpu().numpy())
                # PCA-reducing, whitening and L2-normalization
                numpy_features, _ = preprocess_features(
                    numpy_features, pca_info=cluster_info[1])
                dists = cdist(numpy_features, cluster_info[0], 'euclidean')
                cluster_labels_b = dists.argmin(axis=1)
                cluster_labels_b = torch.from_numpy(
                    cluster_labels_b).unsqueeze(0).to(inputs_ref.device)
                cluster_labels.append(cluster_labels_b)

            ## Finally compute the loss terms due to both images
            ## Ok this is computing a crossentropy directly on the outputs of the net (they are softmaxes, representing soft assignment to every class)
            loss_ref = self.loss(
                inp_ref_flat[b, :, lin_inds_ref].unsqueeze(0), cluster_labels[b])
            loss_other = self.loss(
                inp_other_flat[b, :, lin_inds_other].unsqueeze(0), cluster_labels[b])

            loss += loss_ref
            loss += loss_other

        return loss, cluster_labels
