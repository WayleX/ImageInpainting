import numpy as np
from skimage.color import rgb2gray
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from scipy.ndimage import sobel
from PIL import Image

def main():
    inp = Inpainter(8)

class Inpainter:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def _load_photo(self, path_to_photo):
        self.photo = np.copy(Image.open(path_to_photo))
        self.height, self.width = self.image.shape[:2]

    def _load_mask(self, path_to_mask):
        self.mask = Image.open(path_to_mask)
        
    def _validation(self):
        if self.mask.shape != self.photo[0].shape:
            raise ValueError
        
    def finished(self):
        return np.sum(self.mask) == 0
    
    def update_confidence(self):
        self.front_pos = np.argwhere(self.front == 1)
        new_confidence = np.copy(self.confidence)
        for point in self.front_pos:
            patch = self.get_patch(point)
            new_confidence[*point] = np.sum(self._patch_data(patch)) / self._patch_area()
        self.confidence = new_confidence

    def update_data(self):
        norm = self.get_normal_matrix()
        grad = self.get_gradient_matrix()
        self.data = norm*grad + 0.000001

    def renew_priority(self):
        self.update_confidence()
        self.update_data()
        self.priority = self.confidence * self.data * self.front

    def find_important_region(self):
        pass

    def find_source(self):
        pass

    def update_image(self):
        pass

    def get_normal_matrix(self):
        x_sob = sobel(self.mask, 0)
        y_sob = sobel(self.mask, 1)
        normal = np.dstack((x_sob, y_sob))
        normalize = np.sqrt(y_sob**2 + x_sob**2).repeat(2, axis = 2)
        normalize += 0.00001
        return normal/normalize
    
    def get_gradient_matrix(self):
        grey_img = rgb2gray(self.image)
        grey_img[self.mask == 1] = None
        gradient = np.nan_to_num(np.gradient(grey_img))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros(self.mask.shape)
        for point in self.front_pos:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[*point, 0] = patch_y_gradient[patch_max_pos]
            max_gradient[*point, 1] = patch_x_gradient[patch_max_pos]

        return np.sqrt(max_gradient[:,:,0]**2 + max_gradient[:,:,1]**2)
    
    def get_front(self):
        self.front = laplace(self.mask) > 0

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

    def _get_patch(self, point):
        patch = [
            [max(0, point[0] - np.floor(self.patch_size/2)), min(point[0] + np.ceil(self.patch_size/2), self.height - 1)],
            [max(0, point[1] - np.floor(self.patch_size/2)), min(point[1] + np.ceil(self.patch_size/2), self.width - 1)]
        ]
        return patch

    def _patch_data(self, data, patch):
        return data[patch[0][0]: patch[0][1]+1, patch[1][0]: patch[1][1]+1]
    
    def _patch_area(self, patch):
        return (patch[0][0] - patch[0][1])*(patch[1][0] - patch[1][1])

if __name__=="__main__":
    main()