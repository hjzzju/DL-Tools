from pyxctools.xenocanto import XenoCanto
import argparse
from multiprocessing.pool import ThreadPool
from functools import partial
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="sounds", help="Directory to download to.", required=False)
    args = parser.parse_args()
    xc = XenoCanto()
    page = [i+1 for i in range(300, 15900)]
    with ThreadPool(200) as pool:
        # for i in tqdm():
        #     pass
        pool.map(partial(xc.download_files, search_terms=" ", dir=args.dir), page)
    # for i in range(15900):
    #     print(i)
    #     xc = XenoCanto()
    #     xc.download_files(search_terms=" ", page=i+1, dir=args.dir)
