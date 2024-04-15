import numpy as np
from skimage.color import rgb2gray
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from PIL import Image

def main():
    print('!')

class Inpainter:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def _load_photo(self, path_to_photo):
        self.photo = np.copy(Image.open(path_to_photo))

    def _load_mask(self, path_to_mask):
        self.mask = Image.open(path_to_mask)
        
    def _validation(self):
        if self.mask.shape != self.photo[0].shape:
            raise ValueError
        
    def finished(self):
        return np.sum(self.mask) == 0
    
    def update_confidence(self):
        front_pos = np.argwhere(self.front == 1)
        new_confidence = np.copy(self.confidence)
        for point in front_pos:
            patch = self.get_patch(point)
            # new_confidence[*point] # Do not work on Python < 3.11
            new_confidence[point[0], point[1]] = ... 

    def renew_priority(self):
        self.update_confidence()


    def find_important_region(self):
        pos = np.argmax(self.priority)
        return pos // self.photo.shape[1], pos % self.photo.shape[1]

    def find_source(self):
        pass

    def update_image(self):
        pass

    def get_front(self):
        self.front = laplace(self.mask) > 0

    @staticmethod
    def patch_area(patch):
        return 1 + (patch.shape[0][1] - patch.shape[0][0]) * (patch.shape[1][1] - patch.shape[1][0])

    def get_patch(self, point):
        x_center, y_center = point
        x_start = max(0, x_center - self.patch_size // 2)
        x_end = min(self.photo.shape[1], x_center + self.patch_size // 2)
        y_start = max(0, y_center - self.patch_size // 2)
        y_end = min(self.photo.shape[0], y_center + self.patch_size // 2)

        return self.photo[y_start:y_end, x_start:x_end]

    def inpaint(self, path_to_photo, path_to_mask, path_to_result):
        self._load_photo(path_to_photo)
        self._load_mask(path_to_mask)
        self._validation()
        
        while not self.finished():
            self.get_front()
            self.renew_priority()
            target_region = self.find_important_region()
            source_region = self.find_source(target_region)
            self.update_image(target_region, source_region)
        
        self.photo.save(path_to_result)


if __name__=="__main__":
    main()