from dataset import Google_dataset_of_SIRIWHU_earth_im_tiff, UCMerced_LandUse, CatVsDog, General_template


def create_dataset(name):
    if name == 'Google_dataset_of_SIRIWHU_earth_im_tiff':
        return Google_dataset_of_SIRIWHU_earth_im_tiff.Google_Dataset_of_SIRIWHU
    elif name == 'UCMerced_LandUse':
        return UCMerced_LandUse.UCMercedDataset
    elif name == 'CatVsDog':
        return CatVsDog.CatVsDog
    elif name == 'General_template':
        return General_template.general