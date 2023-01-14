import os
import sys
import subprocess
import numpy as np
import pandas as pd
import sqlite3

from pose_estimation.utils.data_loading import get_ref_2D_data
from pose_estimation.utils.data_loading import get_ground_truth_poses

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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

def discard_imgs_missing_gt( slicepath, query_df ):

    poses, query_names  =   get_ground_truth_poses( query_df.img_name, slicepath )
    query_df.loc[ ~query_df.img_name.isin( query_names ), 'status' ]   =   'missing_gt'
    query_df            =   query_df.merge( pd.DataFrame( { 'img_name': query_names, 'pose': poses } ), on = 'img_name', how = 'left' )

    for q_name in query_df[ query_df.status == 'missing_gt' ].img_name:
        os.remove( os.path.join( slicepath, 'query', q_name ) )

    return query_df

def discard_point_cloud_sparse_areas( slicepath, query_df ):



    return


def summarize( query_df ):
    print( "Tot n queries: ", len( query_df ) )
    print( query_df.groupby( [ 'status', 'side' ] ).img_name.count() )

def main():
    # 1. setup connection to dataset location
    db_url  =   'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/'

    # 2. get slice names, pick only those that are in park
    urls_df =   get_park_slice_urls_list( db_url )

    # Iterate over slices:
    for _, row in urls_df.iterrows():

        # slice_id will be a string like "slice__"
        slice_id    =   row.slice
        slice_num   =   int(slice_id[5:])
        url         =   row.url

        # 3a. load data onto disk
        # 3b. unzip
        # tot ETA: couple hours
        save_slice_data_from_url( url, slice_id )

        # 3c. setup so that all is in place for our verifications
        slicepath   =   os.path.join('', slice_id)
        query_df    =   pd.DataFrame(
            { 'img_name'  :   [ q_name for _, _, q_names in os.walk( os.path.join( slicepath, 'query' ) ) for q_name in q_names ],
              'status'      :   'in_use' } )
        query_df[ 'side' ]  =   np.where( query_df.img_name.str.contains( 'c1' ), 'right', 'left' )

        db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names = get_ref_2D_data( slicepath )

        ref_df              =   pd.DataFrame( { 'img_name': db_image_names, 'image_id': db_image_ids } ).drop_duplicates()
        ref_df              =   ref_df.merge(
            pd.DataFrame( { 'img_name': db_image_names, 'kpt_cnt': 1 } ).groupby( 'img_name' ).kpt_cnt.sum().reset_index(),
            on  = 'img_name',
            how = 'left' )
        ref_df[ 'side' ]    =   np.where( ref_df.img_name.str.contains( 'c1' ), 'right', 'left' )

        # 4. Perform checks and delete all images that did not pass some checks (possibly online)
        #       a.  delete all queries with no ground truth
        query_df    =   discard_imgs_missing_gt( slicepath, query_df )

        #       b.  remove images whose closest reference image sees too few points



        summarize( query_df )




    return

if __name__ == '__main__':
    # test_shell()
    save_slice_data_from_url( 'https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/slice25.tar', 'slice25')

    # main()
