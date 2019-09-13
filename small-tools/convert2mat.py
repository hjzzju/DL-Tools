import scipy.io
import numpy as np

data = np.load("/home/hjz/hjzfolder/vrd-master/pred_res/vrd_pred_roidb_bao0911.npz", allow_pickle=True, encoding="latin1")
scipy.io.savemat('test_pred_bao0911.mat',data)