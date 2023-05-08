import os
import sys
import subprocess
import numpy as np
import pandas as pd
import cv2 as cv
from scipy.spatial import distance_matrix
import time
from datetime import datetime, timedelta, time as dtTime
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from pose_estimation.utils.data_loading import get_camera_parameters,get_ref_2D_data, get_ground_truth_poses, read_gt_file, get_descriptors_image
from fine_grained_segmentation.utils.file_parsing.read_write_model import qvec2rotmat, read_points3D_binary
from utils.utils import img_names_to_datetimes

dirname     =   os.path.dirname( __file__ )


img_height  =   768
img_width   =   1024
gt_repr_err =   5 # pixels
n_NN_flann  =   1

match_scarcity_fine_th      =   7
match_scarcity_med_th       =   14
match_scarcity_coarse_th    =   21

match_scarcity_stable_seq   =   5

short_term_dates            =   [ '20101122', '20101221', '20110304' ]

save_report_name            =   'query_report_2nd_pass.csv'

def test_shell():
    subprocess.run('pwd; cd ../', shell=True)
    subprocess.run('pwd', shell=True, start_new_session=False )

def get_park_slice_urls_list( db_url ):

    urls    =   subprocess.getoutput( 'lynx -dump -listonly ' + db_url ).replace('\n', ' ').split(' ')
    urls    =   [ url for url in urls if url.endswith( '.tar' ) ]

    urls_df             =   pd.DataFrame( { 'url': urls } )
    urls_df[ 'slice' ]  =   urls_df.apply( lambda x: x.url.split('/')[-1][:-4], axis = 1 )
    park_sl             =   [ 'slice' + str(sl_n) for sl_n in range(18, 26)]
    urls_df[ 'type' ]   =   np.where( urls_df['slice'].isin( park_sl ), 'park', 'other' )

    return urls_df[ urls_df[ 'type' ] == 'park' ]

def save_slice_data_from_url( url, slice_id ):
    '''Create a folder with all the slice data from the original datasets'''

    if not os.path.exists( slice_id ):
        cmd_str     =   '''
        wget {url};
        tar -xvf {slice_id}.tar;
        rm {slice_id}.tar
        '''.format(     url=url, slice_id=slice_id )
        subprocess.run( cmd_str, shell=True )

def _update_gt_files( slicepath, delete_names ):

    for root, subdir, files in os.walk(os.path.join(slicepath, 'camera-poses')):
        for f in files:
            filepath    =   os.path.join(root, f)
            with open( filepath, 'r' ) as file:
                lines   =   [ l for l in file.readlines() if not ( l.split()[0] in list( delete_names ) ) ]
            with open( filepath, 'w' ) as file:
                [ file.write( line ) for line in lines ]

def discard_imgs( slicepath, query_df, discard_status ):

    status_flg      =   query_df.status == discard_status

    print( "setup_dataset_match_scarcity@_discard_imgs: Discarding %d images due to %s" %( sum(status_flg), discard_status ) )

    discard_names   =   query_df[ status_flg ].img_name
    for q_name in discard_names:
        os.remove( os.path.join( slicepath, 'query', q_name ) )

    # we will also need to update the GT files deleting what is not there anymore
    print( "\t ... updating ground truth files" )
    _update_gt_files( slicepath, discard_names )


def discard_imgs_missing_gt( slicepath, query_df ):

    poses, query_names  =   get_ground_truth_poses( query_df.img_name, slicepath )
    query_df.loc[ ~query_df.img_name.isin( query_names ), 'status' ]   =   'missing_gt'
    query_df            =   query_df.merge( pd.DataFrame( { 'img_name': query_names, 'pose': poses } ), on = 'img_name', how = 'left' )

    discard_imgs( slicepath, query_df, 'missing_gt' )

    return query_df

def discard_short_term( slicepath, query_df ):
    for root, subdir, files in os.walk( os.path.join( slicepath, 'camera-poses' ) ):
        for f in files:
            if np.any( [ dt in f for dt in short_term_dates ] ):
                filepath = os.path.join( root, f )
                with open( filepath, 'r' ) as file:
                    q_names     =   [ l.split()[0] for l in file.readlines() ]
                query_df.loc[ query_df.img_name.isin( q_names ), 'status' ] =   'short_term'
                for q_name in q_names:
                    os.remove( os.path.join( slicepath, 'query', q_name ) )
                os.remove( filepath )

    return

def _get_close_refs( q_info, ref_poses_df, pos2_th = 25, rot_th = np.cos( np.pi / 6 ) ):

    ref_poses       =   np.stack( ref_poses_df.pose, axis = 0 )
    position_diff2  =   np.sum( ( ref_poses[ :, 4: ] - np.array( q_info.center ) )**2, axis = 1 )
    ref_z_axes      =   np.stack( ref_poses_df.z_axis, axis = 0 )
    q_z_axis        =   q_info.rotmat[ :, -1 ]
    rot_cos         =   ref_z_axes.dot( q_z_axis ) # normalized already, by def of rotation matrix!

    cls_flg         =  ( position_diff2  < pos2_th ) & ( rot_cos > rot_th )

    return ref_poses_df[ cls_flg ].image_id.values

def _get_visible_pt_ids( ref_img_ids, all_ref_img_ids, all_p3D_ids ):

    if len(ref_img_ids) > 5:
        ref_img_ids =   np.sort( ref_img_ids )[::3]
    all_p3D_df  =   pd.DataFrame( { 'image_id': all_ref_img_ids, 'p3D_id': all_p3D_ids } )
    flg         =   all_p3D_df.image_id.isin( ref_img_ids )

    return set( all_p3D_df[ flg ].p3D_id )

def find_match_scarcity( query_df, ref_df, slicepath, slice_num ):

    print( "setup_dataset_match_scarcity@find_match_scarcity:" )
    # Load cameras data
    intrinsics  =   dict()
    dist_coefs  =   dict()
    for camera_id in [ 'c0', 'c1' ]:
        intr, dc    =   get_camera_parameters( camera_id )
        intrinsics[ camera_id ] =   intr
        dist_coefs[ camera_id ] =   dc

    # Load reference data - 2D
    print( "\tGetting reference data from binary file images.bin" )
    db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names = get_ref_2D_data(
        slicepath )

    ref_df = ref_df.merge(
        pd.DataFrame( { 'img_name': db_image_names, 'image_id': db_image_ids, 'kpt_cnt': 1 } ).groupby( [ 'img_name', 'image_id' ] ).kpt_cnt.sum().reset_index(),
        on = 'img_name',
        how = 'left' )
    ref_df[ 'side' ] = np.where( ref_df.img_name.str.contains( 'c1' ), 'right', 'left' )

    # Load reference data - 3D
    print( "\tGetting 3D data from binary file points3D.bin" )
    points3D = read_points3D_binary( os.path.join( slicepath, 'sparse/points3D.bin' ) )
    p_df = pd.DataFrame( points3D ).T.rename( columns = { 1: 'coords' } ).drop( columns = [ 0, 2, 3, 4, 5 ] )
    del points3D

    # Prepare to store useful statistics: number of matches correctly flagged by kNN,
    query_df[ 'num_gt_matches' ]        =   np.nan
    query_df[ 'num_gt_del_clashes' ]    =   np.nan
    query_df[ 'num_kNN_tp_unfiltered' ] =   np.nan
    # query_df[ 'num_kNN_tp_ratiotest' ]  =   np.nan

    # Get nearest reference images for all queries, and point ids for visible points
    query_vis_df    =   query_df[ [ 'img_name', 'pose' ] ].copy()
    ref_vis_df      =   ref_df[ [ 'image_id', 'pose' ] ].copy()

    processing_mins =   round( 160 * len(query_vis_df) / 1000 / 60 )
    print( "\tGetting visible point ids for all query images - we'll be done in {}' approximately, at {:%H:%M}".format( str(processing_mins), datetime.now() + timedelta( minutes = processing_mins ) )  )
    ref_vis_df[ 'z_axis' ]              =   ref_vis_df.apply( lambda x: qvec2rotmat( x.pose[ :4 ] )[:, -1], axis = 1 )
    query_vis_df[ 'rotmat' ]            =   query_vis_df.apply( lambda x: qvec2rotmat( x.pose[ :4 ] ), axis = 1 )
    query_vis_df[ 'center' ]            =   query_vis_df.apply( lambda x: x.pose[ 4: ], axis = 1 )
    query_vis_df[ 'nby_refimg_ids' ]    =   query_vis_df.apply( lambda x: _get_close_refs(x, ref_vis_df), axis = 1 ) # 10s: acceptable
    query_vis_df[ 'visible_p3D_ids' ]   =   query_vis_df.apply( lambda x: list( _get_visible_pt_ids(x.nby_refimg_ids, db_image_ids, db_p3D_ids) ), axis = 1 ) # Est 10 min

    st              =   time.time()

    # Check if any query_df was previously saved, and if so load the results to avoid recomputing
    if os.path.exists( os.path.join( slicepath, 'reports', save_report_name ) ):
        query_df    =   pd.read_csv( os.path.join( slicepath, 'reports', save_report_name ) )
    saved_indices   =   query_df.dropna( subset = [ 'num_kNN_tp_unfiltered' ] ).index

    # For each image, project selected points, retrieve keypoints, match them and compare to kNN matches obtained via F
    for c, (rownum, query) in enumerate( query_vis_df.iterrows() ):

        if rownum in saved_indices:
            print( "\t\tUsing previously saved data for query {} out of {} ".format( c+1, len( query_vis_df )) )
            continue

        print( "\t\tEvaluating GT matches for query {} out of {} ".format( c+1, len( query_vis_df )) )

        query_name          =   query.img_name
        camera_id           =   'c0' if 'c0' in query_name else 'c1'
        camera_matrix       =   np.array( intrinsics[ camera_id ] )
        distortion_coefs    =   np.array( dist_coefs[ camera_id ] )

        R_gt_query          =   query.rotmat
        c_gt_query          =   np.array(query.center)
        t_gt_query          =   - R_gt_query.dot( c_gt_query )
        rodrigues_gt_query  =   cv.Rodrigues(R_gt_query)[0]

        # Project points
        p3D                 =   np.stack( p_df.loc[ query.visible_p3D_ids, 'coords' ], axis = 0 )
        p2D                 =   cv.projectPoints( p3D, rodrigues_gt_query, t_gt_query, camera_matrix, distortion_coefs )[0].reshape( ( -1, 2 ) )
        p2D                 =   np.rint( p2D ).astype( np.int16 )

        range_flg           =   ( ( p2D > np.zeros_like( p2D ) ) & (
            np.append( np.ones( (len( p2D ), 1) ) * img_width, np.ones( (len( p2D ), 1) ) * img_height, axis = 1 ) > p2D ) ).all(
            axis = 1 )

        p2D                     =   p2D[ range_flg ]
        p3D_ids                 =   np.array( query.visible_p3D_ids )[ range_flg ]

        qkp, q_descriptors      =   get_descriptors_image( query_name, slicepath, 'query', slice = slice_num )
        qkp_ids                 =   np.arange( len(qkp) )
        qkp                     =   qkp[:, :2]

        # Find GT couplings
        gt_dist_mat             =   distance_matrix( qkp, p2D )
        gt_match_flg            =   np.where( gt_dist_mat < gt_repr_err )
        gt_match_df             =   pd.DataFrame( gt_match_flg, index = [ 'q_id', 'point_cloud' ] ).T
        # gt_match_df[ 'q_id' ]   =   qkp_ids[ gt_match_flg[ 0 ] ]
        gt_match_df[ 'pc_id' ]  =   p3D_ids[ gt_match_flg[ 1 ] ]

        # Address potentially arising double associations of a query keypoint to point cloud keypoints (PC keypoints are assumed to be distinct points)
        gt_match_df[ 'n_corr_pq' ]                  =   gt_match_df.groupby( 'q_id' )[ 'pc_id' ].transform( 'count' )

        conflict_matches_flg                        =   gt_match_df.n_corr_pq > 1
        if sum( conflict_matches_flg ):
            conflict_matches_df                     =   gt_match_df[ conflict_matches_flg ].copy()
            conflict_matches_df[ 'dist_from_camera' ] = distance_matrix(
                np.stack( p_df.loc[ conflict_matches_df.pc_id, 'coords' ] ), c_gt_query.reshape( 1, -1 ) ).flatten()
            conflict_matches_df[ 'is_closest_match' ] = conflict_matches_df.groupby( 'q_id' )['dist_from_camera' ].transform(
                'min' ) == conflict_matches_df.dist_from_camera
            gt_match_df                             =   gt_match_df.merge( conflict_matches_df[ [ 'q_id', 'pc_id', 'is_closest_match' ] ], on = [ 'q_id', 'pc_id' ], how = 'left' )
            gt_match_df[ 'is_closest_match' ]       =   gt_match_df.is_closest_match.fillna( True )
        else:
            gt_match_df[ 'is_closest_match' ]       =   True

        gt_match_df                                 =   gt_match_df[ gt_match_df.is_closest_match ]
        # note: oftentimes the two or more competing 3D points are quite close in space (from cm to m). Is it still correct to discard one of them? wouldn't we want to keep them both perhaps, if very close?

        # Find kNN couplings
        flann_matcher       =   cv.FlannBasedMatcher( dict( algorithm = 1, trees = 5 ), dict( checks = 50 ) )
        k_nearest_matches   =   flann_matcher.knnMatch( q_descriptors.astype( np.float32 ),
                                                         db_descriptors.astype( np.float32 ), k = n_NN_flann )

        kNN_couplings       =   []
        for ms in k_nearest_matches:
            for m in ms:
                kNN_couplings.append( [ m.queryIdx, db_p3D_ids[ m.trainIdx ] ] )

        kNN_match_df        =   pd.DataFrame( kNN_couplings, columns = [ 'q_id', 'pc_id' ] )
        kNN_match_df[ 'found_by_kNN' ] =   True

        # Compare: how many are there?
        img_match_df        =   pd.merge( gt_match_df, kNN_match_df, on = [ 'q_id', 'pc_id' ], how = 'left' )

        n_tp_matches        =   img_match_df.found_by_kNN.fillna(False).sum()

        query_df.loc[ rownum, 'num_gt_matches' ]        =   len( gt_match_df )
        query_df.loc[ rownum, 'num_gt_del_clashes']     =   ( ~conflict_matches_df.is_closest_match ).sum()
        query_df.loc[ rownum, 'num_kNN_tp_unfiltered' ] =   n_tp_matches
        # query_df.loc[ rownum, 'num_kNN_tp_ratiotest' ]  =   np.nan

        if n_tp_matches >   match_scarcity_coarse_th:
            print( "\t\tQuery {} will be removed as there were enough matches.".format(query_name) )
            query_df.loc[ rownum, 'status' ]    =   'no_match_scarcity'

        print( "\t\tElapsed time: {}".format( timedelta( seconds = time.time() - st ) ) )

        if (c+1)%100 == 0:
            print( "\t\tSaving intermediate results on match scarcity analysis" )
            query_df.to_csv( os.path.join( slicepath, 'reports', save_report_name ) )

    return query_df, ref_df

def find_stable_match_scarcity( query_df ):

    # 1. count length sequence with match scarcity
        # group by date and side, sort by time
    query_consec_frames                         =   query_df.sort_values( 'datetime' ).copy()
    query_consec_frames[ 'date' ]               =   query_consec_frames.datetime.dt.date
        # group so that every last consecutive pic with match scarcity is the last in the group
    query_consec_frames[ 'consec_group_id' ]    =   query_consec_frames.groupby( [ 'date', 'side' ], as_index = False, group_keys = False).apply( lambda df: ( df.status != df.status.shift() ).cumsum() )
    query_consec_frames                         =   pd.merge(
        query_consec_frames.reset_index(),
        query_consec_frames.groupby(['date', 'consec_group_id']).agg( seq_len = ('img_name', 'count')).reset_index(),
        how = 'left',
        on = [ 'date', 'consec_group_id' ]
    ).set_index('index')

    # 2. discard all match scarcity that lasts fewer frames
    flg_ms                                              =   query_consec_frames.status == 'in_use'
    flg_stable                                          =   query_consec_frames.seq_len > match_scarcity_stable_seq
    query_df.loc[ flg_ms & (~flg_stable), 'status' ]    =   'no_stable_match_scarcity'

    return query_df

def discard_match_scarcity( slicepath, query_df ):
    discard_imgs( slicepath, query_df, 'no_match_scarcity' )
    discard_imgs( slicepath, query_df, 'no_stable_match_scarcity' )

def summarize( query_df ):
    print( "Tot n queries: ", len( query_df ) )
    print( query_df.groupby( [ 'status', 'side' ] ).img_name.count() )

def main():
    # 1. setup connection to dataset location
    db_url  =   'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/'

    # 2. get slice names, pick only those that are in park
    urls_df =   get_park_slice_urls_list( db_url )
    # urls_df =   get_non_park_slice_urls_list( db_url )

    # Iterate over slices:
    for _, row in urls_df.tail(1).iterrows():

        # slice_id will be a string like "slice__"
        slice_id    =   row.slice
        slice_num   =   int(slice_id[5:])
        url         =   row.url

        # 3a. load data onto disk
        # 3b. unzip
        # tot ETA: couple hours
        # save_slice_data_from_url( url, slice_id )

        # 3c. setup so that all is in place for our verifications
        slicepath   =   os.path.join(dirname, slice_id)
        query_df    =   pd.DataFrame(
            { 'img_name'        :   [ q_name for _, _, q_names in os.walk( os.path.join( slicepath, 'query' ) ) for q_name in q_names ],
              'status'          :   'in_use' } )
        query_df[ 'side' ]      =   np.where( query_df.img_name.astype("string").str.contains( 'c1' ), 'right', 'left' )
        query_df[ 'datetime' ]  =   img_names_to_datetimes( query_df.img_name )

        if not os.path.exists( os.path.join( slicepath, 'reports' ) ):
            os.mkdir(                   os.path.join( slicepath, 'reports' ) ) # setup directory for loading the

        ref_gt_filepath         =   os.path.join( slicepath, 'ground-truth-database-images-{}.txt'.format(slice_id) )
        ref_pose_dict           =   read_gt_file( ref_gt_filepath )

        ref_df                  =   pd.DataFrame( { 'img_name': ref_pose_dict.keys(), 'pose': ref_pose_dict.values() } )

        # 4. Perform checks and delete all images that did not pass some checks (possibly online)
        #       a.  delete all queries with no ground truth
        query_df                =   discard_imgs_missing_gt( slicepath, query_df )

        #       b.  remove images whose closest reference image sees too few points - not really the case anywhere by looking at the numbers
        #       don't think that's an issue. Could instead remove all sequences that were taken too close in time to ref

        #       c.  cherry-pick match scarcity
        query_df, ref_df        =   find_match_scarcity( query_df, ref_df, slicepath, slice_num )
        #           make it stable
        query_df                =   find_stable_match_scarcity( query_df )

        # discard_match_scarcity( slicepath, query_df )


        query_df.to_csv( os.path.join( slicepath, 'reports', save_report_name ) )
        # save it instead

        summarize( query_df )

        # First attempt one slice only
        break

    return

if __name__ == '__main__':
    # test_shell()
    # save_slice_data_from_url( 'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/slice25.tar', 'slice25')

    main()
