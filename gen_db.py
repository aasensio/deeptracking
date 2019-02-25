import numpy as np
import h5py
from astropy.io import fits
import glob
from tqdm import tqdm
from ipdb import set_trace as stop
from skimage.feature import register_translation

datasets = ['continuum4170','Gband','CaK','Mgb2'] #,'Halpha']
pixel_size = [0.18, 0.18, 0.18, 0.18, 0.09]
cadences = [2.11*14, 2.11*14, 2.11*14, 4.22*7, 0.99*30]

def generate_dataset(output_file, n_patches_per_image, n_images, border=200, n_pixel=64):

    n_patches = n_images * n_patches_per_image

    print("Generating file {0}...".format(output_file))
    print(" - N. images : {0}".format(n_images))
    print(" - N. patches per image : {0}".format(n_patches_per_image))
    print(" - N. patches total : {0}".format(n_patches))
    
    index_datasets = np.random.randint(low=0, high=len(datasets), size=n_images)

    files = []

    for ds in datasets:
        tmp = glob.glob('/net/nas/proyectos/fis/aasensio/deep_learning/deepvel_jess/{0}/*.fits'.format(ds))
        tmp.sort()
        files.append(tmp)

    f = h5py.File('/net/nas/proyectos/fis/aasensio/deep_learning/deepvel_jess/{0}'.format(output_file), 'w')
    db_im = f.create_dataset("images", (n_patches, 2, n_pixel, n_pixel), dtype=np.float32)

    loop = 0

    for i in tqdm(range(n_images)):
        ds_files = files[index_datasets[i]]
        index = np.random.randint(low=0, high=len(ds_files)-1, size=1)[0]

        x0 = np.random.randint(low=border, high=1000-border, size=n_patches_per_image)
        y0 = np.random.randint(low=border, high=1000-border, size=n_patches_per_image)

        f0 = fits.open(ds_files[index])
        f1 = fits.open(ds_files[index+1])

        for j in range(n_patches_per_image):

            im0 = f0[0].data[x0[j]:x0[j]+n_pixel,y0[j]:y0[j]+n_pixel]
            im1 = f1[0].data[x0[j]:x0[j]+n_pixel,y0[j]:y0[j]+n_pixel]

            shift, error, diffphase = register_translation(im0, im1)

            shift = [int(f) for f in shift]
                            
            im1 = np.roll(im1, shift, axis=(0,1))

            db_im[loop,0,:,:] = im0
            db_im[loop,1,:,:] = im1

            loop += 1

        f0.close()
        f1.close()
        
    f.close()

if (__name__ == '__main__'):
    # n_patches_per_image = 10
    # n_images = 5000
    # border = 200
    # n_pixel = 64

    # generate_dataset('training.h5', n_patches_per_image, n_images, border, n_pixel)


    n_patches_per_image = 10
    n_images = 500
    border = 200
    n_pixel = 64

    generate_dataset('validation.h5', n_patches_per_image, n_images, border, n_pixel)