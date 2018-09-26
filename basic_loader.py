import numpy as np
import cv2
import os
from collections import namedtuple
Batch = namedtuple('Batch', ['data', 'label', 'pad'])


def load_batch(src_img_seg, crop_size, is_train, shrink_size, pad):
    hc, wc = crop_size
    batch_img, batch_seg = [], []
    scale_pool = [0.5, 0.75, 1, 1.25, 1.5]
    for src_img, src_seg in src_img_seg:
        img = cv2.imread(src_img)
        seg = cv2.imread(src_seg, 0)
        assert img.shape[:2] == seg.shape[:2]
        h, w = img.shape[:2]

        if is_train:
            # random mirror
            if np.random.rand() > 0.5:
                img = img[:, ::-1]
                seg = seg[:, ::-1]

            # random scale
            rand_scale = scale_pool[np.random.randint(0, 5)]
            if rand_scale != 1:
                h = int(h * rand_scale + .5)
                w = int(w * rand_scale + .5)
                img = cv2.resize(img, (w, h))
                seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

        pad_h = max(hc - h, 0)
        pad_w = max(wc - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, 127)
            seg = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, 255)
            h, w = img.shape[:2]
        
        if is_train:
            h_off = np.random.randint(0, h - hc + 1)
            w_off = np.random.randint(0, w - wc + 1)
        else:
            h_off = (h - hc) // 2
            w_off = (w - wc) // 2

        img = img[h_off:h_off+hc, w_off:w_off+wc]
        seg = seg[h_off:h_off+hc, w_off:w_off+wc]

        if shrink_size is not None:
            seg = cv2.resize(seg, shrink_size)

        batch_img.append(img)
        batch_seg.append(seg)

    # Mean BGR: 104.008, 116.669, 122.675
    # batch_img.shape = (N, C, H, W), RGB color (NOT BGR!)
    batch_img = np.array(batch_img)[..., ::-1].transpose(0, 3, 1, 2) - \
                np.array([122.675, 116.669, 104.008]).reshape(1, 3, 1, 1)
    
    # batch_seg.shape = (N, H, W)
    batch_seg = np.array(batch_seg)

    # Pad to reach the target batch_size
    if pad > 0:
        seg_size = batch_seg.shape[1:]
        batch_img = np.concatenate([batch_img, np.ones((pad, 3, hc, wc)) * 127], axis=0)
        batch_seg = np.concatenate([batch_seg, np.ones((pad,) + seg_size) * 255], axis=0)

    return Batch(data=batch_img, label=batch_seg, pad=pad)


class VOCSegLoader(object):
    def __init__(self, image_root, label_root, data_list, batch_size, data_shape, is_train,
                 shrink_size=None, pad=False, shuffle=False):
        '''
        image_root:  /some/path/to/VOC2012/JPEGImages
                     The folder of .jpg images.
        label_root:  /some/path/to/SegmentationClassAug
                     The folder of .png labels, each pixel valued from 0 to 
                     NUM_CLS-1, (or ignored 255)
        data_list:   /the/path/to/train_aug.txt (or val.txt)
        data_shape:  (3, H, W), input shape, e.g. (3, 321, 321) for deeplab-v2
        is_train:    bool, use random_crop+random_mirror+random_scale if True,
                           use center_crop if False
        shrink_size: (H, W), target label shape, e.g. (41, 41) for deeplab-v2
        pad:         bool, whether to fill the batch with dummy image&label
                           to ensure the target batch_size if there are not
                           ample samples.
        shuffle:     bool, whether to shuffle the order of image list randomly
        '''
        # data_name = [x.strip() for x in file(data_list).readlines()]

        with open(data_list) as _f:
            data_name = [x.strip() for x in _f.readlines()]

        image_list = [os.path.join(image_root, x + '.jpg') for x in data_name]
        label_list = [os.path.join(label_root, x + '.png') for x in data_name]

        self.image_list = image_list
        self.label_list = label_list
        self.index_list = np.arange(len(self.image_list))
        self.shrink_size = tuple(shrink_size) if shrink_size is not None else None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_shape = tuple(data_shape)
        self.is_train = is_train
        self.pad = pad
        if self.pad:
            self.num_batch = (len(self.image_list) + self.batch_size - 1) // self.batch_size
        else:
            self.num_batch = len(self.image_list) // self.batch_size

        self.reset()

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.index_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.num_batch:
            raise StopIteration
        indices = self.index_list[self.current*self.batch_size : (self.current+1)*self.batch_size]
        self.current += 1

        src_img_seg = [_x for _x in zip([self.image_list[i] for i in indices],
                            [self.label_list[i] for i in indices])]
        pad = self.batch_size - len(src_img_seg)

        batch = load_batch(src_img_seg, self.data_shape[1:], self.is_train,
                           self.shrink_size, pad)
        return batch

    # for python2
    next = __next__

    #TODO: multi-processing version



if __name__ == '__main__':
    # Usage example
    image_root = '/some/path/to/VOC2012/JPEGImages'
    label_root = '/some/path/to/SegmentationClassAug'
    data_list = './data/train_aug.txt'
    batch_size = 10
    data_shape = (3, 321, 321)
    shrink_size = (41, 41)
    is_train= True

    # make loader
    loader = VOCSegLoader(image_root, label_root, data_list, batch_size, data_shape, is_train, shrink_size)

    # run
    for i, batch in enumerate(loader):
        print(i, batch.data.shape, batch.label.shape, batch.pad, batch.data.sum())

    # reset for next epoch
    loader.reset()

    # run
    for i, batch in enumerate(loader):
        print(i, batch.data.shape, batch.label.shape, batch.pad, batch.data.sum())
