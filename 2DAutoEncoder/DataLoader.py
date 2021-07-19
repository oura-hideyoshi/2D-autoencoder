import numpy as np
from pathlib import Path
from cv2 import imread
from cv2 import IMREAD_GRAYSCALE
from numpy import expand_dims
from scipy.io import loadmat, savemat
import os


class DataLoader:
    def __init__(self, data_path, im_dim):
        """set train_path

        Parameters
        ----------
        data_path: str
            dataset path.
            (expected: receive from args.train_noise_path or args.train_real_path)

        """
        self.path = data_path
        self.im_dim = im_dim
        # self.dataset = args.dataset

    def load_data(self):
        """load data from data_path.

        Returns
        -------
        data_set: ndarray
            ndarray of 2d dataset. (index, dim, dim, dim)

        File Structure
        -------
        data_path
        ┣image1.tif
        ┣image2.tif

        Notes
        -------
        * load images as 'GRAY SCALE'
        """
        data_set = []
        for file in Path(self.path).glob("*.tiff"):
            # print('Loading image... ' + str(file))
            img = imread(str(file), IMREAD_GRAYSCALE)
            img = expand_dims(img, axis=2)
            if img.shape != self.im_dim:
                print("LOADED IMAGE SHAPE :", img.shape, "SAT im_dim at cfg :", self.im_dim)
                raise ValueError("Loaded data shape doesn't matches.")
            data_set.append(img)

        print("Loaded data from", os.path.abspath(self.path))
        return data_set
        # get file length
    # raw = np.load(self.train_path+'/'+self.dataset+'.mat')
    # print('Loaded data with '+str(raw.shape[0])+'objects')


if __name__ == "__main__":
    dataloader = DataLoader("../dataset2D/ReconData5set/256x256pix/train/image", (256,256,1))
    data = dataloader.load_data()
    pass
