import pandas as pd

def img_names_to_datetimes( img_names ):
    timestamps  =   img_names.apply( lambda x: int( x.split('_')[-1][:-6] ) )
    return pd.to_datetime( timestamps, unit = 'us' )

