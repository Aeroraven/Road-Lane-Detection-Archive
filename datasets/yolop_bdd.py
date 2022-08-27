import json

from tqdm import tqdm

from datasets.yolop_abstract import AutoDriveDataset


class BddDataset(AutoDriveDataset):
    def __init__(self,
                 train_path,
                 mask_path,
                 is_train,
                 inputsize,
                 transform=None):
        super().__init__(train_path,mask_path, is_train, inputsize, transform)
        self.db = self._get_db()

    def _get_db(self):
        print('building database...')
        gt_db = []
        for mask in tqdm(list(self.mask_list),ascii=True):
            mask_path = str(mask)
            image_path = mask_path.replace(str(self.lane_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.lane_root), str(self.lane_root))
            rec = [{
                'image': image_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        pass