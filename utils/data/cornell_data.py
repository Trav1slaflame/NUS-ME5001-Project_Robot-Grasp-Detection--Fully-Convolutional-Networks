import os
import glob

from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        # # Used for training!
        # depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        # rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        # Used for testing!
        # depthf = ['data/captured_img/0.tiff', 'data/captured_img/1.tiff', 'data/captured_img/adv_1.tiff', 'data/captured_img/adv_2.tiff', 'data/captured_img/adv_3.tiff', 'data/captured_img/adv_4.tiff', 'data/captured_img/adv_5.tiff', 'data/captured_img/adv_6.tiff', 'data/captured_img/adv_7.tiff', 'data/captured_img/adv_c_1.tiff', 'data/captured_img/adv_c_2.tiff', 'data/captured_img/b_1.tiff', 'data/captured_img/b_2.tiff', 'data/captured_img/b_3.tiff', 'data/captured_img/b_4.tiff', 'data/captured_img/b_5.tiff', 'data/captured_img/b_6.tiff', 'data/captured_img/c_1.tiff', 'data/captured_img/c_2.tiff', 'data/captured_img/c_3.tiff']
        # rgbf = ['data/captured_img/0.png', 'data/captured_img/1.png', 'data/captured_img/adv_1.png', 'data/captured_img/adv_2.png', 'data/captured_img/adv_3.png', 'data/captured_img/adv_4.png', 'data/captured_img/adv_5.png', 'data/captured_img/adv_6.png', 'data/captured_img/adv_7.png', 'data/captured_img/adv_c_1.png', 'data/captured_img/adv_c_2.png', 'data/captured_img/b_1.png', 'data/captured_img/b_2.png', 'data/captured_img/b_3.png', 'data/captured_img/b_4.png', 'data/captured_img/b_5.png', 'data/captured_img/b_6.png', 'data/captured_img/c_1.png', 'data/captured_img/c_2.png', 'data/captured_img/c_3.png']

        # depthf = ['data/captured_img/0.tiff', 'data/captured_img/1.tiff', 'data/captured_img/2.tiff', 'data/captured_img/3.tiff', 'data/captured_img/4.tiff', 'data/captured_img/5.tiff', 'data/captured_img/6.tiff', 'data/captured_img/7.tiff', 'data/captured_img/8.tiff', 'data/captured_img/9.tiff', 'data/captured_img/10.tiff', 'data/captured_img/11.tiff', 'data/captured_img/12.tiff', 'data/captured_img/13.tiff', 'data/captured_img/14.tiff', 'data/captured_img/15.tiff', 'data/captured_img/16.tiff', 'data/captured_img/17.tiff', 'data/captured_img/18.tiff', 'data/captured_img/19.tiff']
        # rgbf = ['data/captured_img/0.png', 'data/captured_img/1.png', 'data/captured_img/2.png', 'data/captured_img/3.png', 'data/captured_img/4.png', 'data/captured_img/5.png', 'data/captured_img/6.png', 'data/captured_img/7.png', 'data/captured_img/8.png', 'data/captured_img/9.png', 'data/captured_img/10.png', 'data/captured_img/11.png', 'data/captured_img/12.png', 'data/captured_img/13.png', 'data/captured_img/14.png', 'data/captured_img/15.png', 'data/captured_img/16.png', 'data/captured_img/17.png', 'data/captured_img/18.png', 'data/captured_img/19.png']

        # depthf = ['data/captured_img/20.tiff', 'data/captured_img/21.tiff', 'data/captured_img/22.tiff', 'data/captured_img/23.tiff', 'data/captured_img/24.tiff', 'data/captured_img/25.tiff', 'data/captured_img/26.tiff', 'data/captured_img/27.tiff', 'data/captured_img/28.tiff', 'data/captured_img/29.tiff', 'data/captured_img/30.tiff', 'data/captured_img/31.tiff', 'data/captured_img/32.tiff', 'data/captured_img/33.tiff', 'data/captured_img/34.tiff', 'data/captured_img/35.tiff', 'data/captured_img/36.tiff', 'data/captured_img/37.tiff', 'data/captured_img/38.tiff', 'data/captured_img/39.tiff']
        # rgbf = ['data/captured_img/20.png', 'data/captured_img/21.png', 'data/captured_img/22.png', 'data/captured_img/23.png', 'data/captured_img/24.png', 'data/captured_img/25.png', 'data/captured_img/26.png', 'data/captured_img/27.png', 'data/captured_img/28.png', 'data/captured_img/29.png', 'data/captured_img/30.png', 'data/captured_img/31.png', 'data/captured_img/32.png', 'data/captured_img/33.png', 'data/captured_img/34.png', 'data/captured_img/35.png', 'data/captured_img/36.png', 'data/captured_img/37.png', 'data/captured_img/38.png', 'data/captured_img/39.png']
        
        # Haozhe's captured images
        # depthf = ['data/test/1.tiff','data/test/2.tiff','data/test/3.tiff','data/test/4.tiff','data/test/5.tiff','data/test/6.tiff','data/test/7.tiff',
        # 'data/test/8.tiff','data/test/9.tiff','data/test/10.tiff','data/test/11.tiff','data/test/12.tiff','data/test/13.tiff','data/test/14.tiff','data/test/15.tiff',
        # 'data/test/16.tiff','data/test/17.tiff','data/test/18.tiff','data/test/19.tiff','data/test/20.tiff','data/test/21.tiff', 'data/test/22.tiff']

        # rgbf = ['data/test/1.png','data/test/2.png','data/test/3.png','data/test/4.png','data/test/5.png','data/test/6.png','data/test/7.png',
        # 'data/test/8.png','data/test/9.png','data/test/10.png','data/test/11.png','data/test/12.png','data/test/13.png','data/test/14.png','data/test/15.png',
        # 'data/test/16.png','data/test/17.png','data/test/18.png','data/test/19.png','data/test/20.png','data/test/21.png', 'data/test/22.png']

        # depthf = ['data/test_2/0.tiff','data/test_2/2.tiff','data/test_2/3.tiff','data/test_2/4.tiff','data/test_2/5.tiff','data/test_2/6.tiff','data/test_2/7.tiff',
        # 'data/test_2/8.tiff','data/test_2/9.tiff','data/test_2/10.tiff','data/test_2/11.tiff','data/test_2/12.tiff','data/test_2/13.tiff','data/test_2/14.tiff','data/test_2/15.tiff',
        # 'data/test_2/16.tiff','data/test_2/17.tiff','data/test_2/18.tiff','data/test_2/19.tiff','data/test_2/20.tiff','data/test_2/21.tiff']

        # rgbf = ['data/test_2/0.png','data/test_2/2.png','data/test_2/3.png','data/test_2/4.png','data/test_2/5.png','data/test_2/6.png','data/test_2/7.png',
        # 'data/test_2/8.png','data/test_2/9.png','data/test_2/10.png','data/test_2/11.png','data/test_2/12.png','data/test_2/13.png','data/test_2/14.png','data/test_2/15.png',
        # 'data/test_2/16.png','data/test_2/17.png','data/test_2/18.png','data/test_2/19.png','data/test_2/20.png','data/test_2/21.png']

        # depthf = ['data/test_3/0.tiff','data/test_3/1.tiff','data/test_3/2.tiff','data/test_3/3.tiff','data/test_3/4.tiff','data/test_3/5.tiff','data/test_3/6.tiff',
        # 'data/test_3/7.tiff','data/test_3/8.tiff','data/test_3/9.tiff','data/test_3/10.tiff','data/test_3/11.tiff','data/test_3/12.tiff','data/test_3/13.tiff','data/test_3/14.tiff',
        # 'data/test_3/15.tiff','data/test_3/16.tiff','data/test_3/17.tiff','data/test_3/18.tiff','data/test_3/19.tiff','data/test_3/20.tiff', 'data/test_3/21.tiff',
        # 'data/test_3/22.tiff','data/test_3/24.tiff','data/test_3/25.tiff','data/test_3/26.tiff','data/test_3/27.tiff','data/test_3/28.tiff','data/test_3/29.tiff','data/test_3/30.tiff',
        # 'data/test_3/31.tiff','data/test_3/32.tiff','data/test_3/33.tiff','data/test_3/34.tiff','data/test_3/35.tiff','data/test_3/36.tiff','data/test_3/37.tiff','data/test_3/38.tiff',
        # 'data/test_3/39.tiff','data/test_3/40.tiff','data/test_3/41.tiff','data/test_3/42.tiff','data/test_3/43.tiff','data/test_3/44.tiff','data/test_3/45.tiff','data/test_3/46.tiff',
        # 'data/test_3/47.tiff','data/test_3/48.tiff','data/test_3/49.tiff','data/test_3/50.tiff','data/test_3/51.tiff','data/test_3/52.tiff','data/test_3/53.tiff','data/test_3/54.tiff',
        # 'data/test_3/55.tiff','data/test_3/56.tiff','data/test_3/57.tiff','data/test_3/58.tiff','data/test_3/59.tiff','data/test_3/60.tiff','data/test_3/61.tiff','data/test_3/62.tiff', 
        # 'data/test_3/63.tiff','data/test_3/64.tiff','data/test_3/65.tiff','data/test_3/66.tiff','data/test_3/67.tiff','data/test_3/68.tiff','data/test_3/69.tiff','data/test_3/70.tiff',
        # 'data/test_3/71.tiff','data/test_3/72.tiff','data/test_3/73.tiff','data/test_3/74.tiff',]

        # rgbf = ['data/test_3/0.png','data/test_3/1.png','data/test_3/2.png','data/test_3/3.png','data/test_3/4.png','data/test_3/5.png','data/test_3/6.png',
        # 'data/test_3/7.png','data/test_3/8.png','data/test_3/9.png','data/test_3/10.png','data/test_3/11.png','data/test_3/12.png','data/test_3/13.png','data/test_3/14.png',
        # 'data/test_3/15.png','data/test_3/16.png','data/test_3/17.png','data/test_3/18.png','data/test_3/19.png','data/test_3/20.png', 'data/test_3/21.png',
        # 'data/test_3/22.png','data/test_3/24.png','data/test_3/25.png','data/test_3/26.png','data/test_3/27.png','data/test_3/28.png','data/test_3/29.png','data/test_3/30.png',
        # 'data/test_3/31.png','data/test_3/32.png','data/test_3/33.png','data/test_3/34.png','data/test_3/35.png','data/test_3/36.png','data/test_3/37.png','data/test_3/38.png',
        # 'data/test_3/39.png','data/test_3/40.png','data/test_3/41.png','data/test_3/42.png','data/test_3/43.png','data/test_3/44.png','data/test_3/45.png','data/test_3/46.png',
        # 'data/test_3/47.png','data/test_3/48.png','data/test_3/49.png','data/test_3/50.png','data/test_3/51.png','data/test_3/52.png','data/test_3/53.png','data/test_3/54.png',
        # 'data/test_3/55.png','data/test_3/56.png','data/test_3/57.png','data/test_3/58.png','data/test_3/59.png','data/test_3/60.png','data/test_3/61.png','data/test_3/62.png', 
        # 'data/test_3/63.png','data/test_3/64.png','data/test_3/65.png','data/test_3/66.png','data/test_3/67.png','data/test_3/68.png','data/test_3/69.png','data/test_3/70.png',
        # 'data/test_3/71.png','data/test_3/72.png','data/test_3/73.png','data/test_3/74.png',]

        depthf = ['data/test_4/0.tiff', 'data/test_4/1.tiff', 'data/test_4/2.tiff', 'data/test_4/3.tiff', 'data/test_4/4.tiff', 'data/test_4/5.tiff', 'data/test_4/6.tiff', 'data/test_4/7.tiff', 
        'data/test_4/8.tiff', 'data/test_4/9.tiff', 'data/test_4/10.tiff', 'data/test_4/11.tiff', 'data/test_4/12.tiff', 'data/test_4/13.tiff', 'data/test_4/14.tiff', 'data/test_4/15.tiff', 
        'data/test_4/16.tiff', 'data/test_4/21.tiff', 'data/test_4/22.tiff', 'data/test_4/23.tiff', 'data/test_4/24.tiff', 'data/test_4/25.tiff', 'data/test_4/26.tiff', 'data/test_4/27.tiff', 
        'data/test_4/28.tiff']
        rgbf = ['data/test_4/0.png', 'data/test_4/1.png', 'data/test_4/2.png', 'data/test_4/3.png', 'data/test_4/4.png', 'data/test_4/5.png', 'data/test_4/6.png', 'data/test_4/7.png', 'data/test_4/8.png', 
        'data/test_4/9.png', 'data/test_4/10.png', 'data/test_4/11.png', 'data/test_4/12.png', 'data/test_4/13.png', 'data/test_4/14.png', 'data/test_4/15.png', 'data/test_4/16.png', 'data/test_4/21.png',
        'data/test_4/22.png', 'data/test_4/23.png', 'data/test_4/24.png', 'data/test_4/25.png', 'data/test_4/26.png', 'data/test_4/27.png', 'data/test_4/28.png']

        # depthf = ['data/test_5/1.tiff', 'data/test_5/2.tiff','data/test_5/3.tiff']
        # rgbf = ['data/test_5/1.png', 'data/test_5/2.png','data/test_5/3.png']


        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
