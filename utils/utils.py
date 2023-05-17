import pandas as pd
import time

def img_names_to_datetimes( img_names ):
    timestamps  =   img_names.apply( lambda x: int( x.split('_')[-1][:-6] ) )
    return pd.to_datetime( timestamps, unit = 'us' )

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def my_timeit( fun ):
    def wrap_func( *args, **kwargs ):
        ts = time.time()
        res = fun( *args, **kwargs )
        print( f'Function {fun.__name__!r} executed in {(time.time()-ts):.6f}s' )
        return res
    return wrap_func
