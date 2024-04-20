import sys
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage import convolve
from scipy.ndimage import sobel
from PIL import Image

class Inpainter():
    def __init__(self, src: str):
        self.__image = self.__load_image(src)
        
        self.__width = self.__image.shape[1]
        self.__height = self.__image.shape[0]

        self.__mask = None
        self.__front = None
        self.__data = None
        self.__confidence = None
        self.__priority = None

        self.__patch_size = None

    def __load_image(self, src: str):
        return np.array(Image.open(src))

    def __inpaint(self):
        while not self.__check_full_coverage():
            # image_prev = np.copy(self.__image)
            self.__update_front()
            self.__update_priority()
            
            hp_patch_center = self.__highest_priority_patch_center()
            new_patch = self.__best_patch(hp_patch_center)

            self.__update(hp_patch_center, new_patch)

            print(np.sum(self.__mask))
            # print(f"Progress: {np.sum(self.__mask) / (self.__width * self.__height) * 100}%")

    def __update(self, hp_pixel, new_patch):
        hp_patch_coords = self.__patch_coords_by_center(hp_pixel)

        update_positions = np.argwhere(self.__patch_data(self.__mask, hp_patch_coords) == 1) + [hp_patch_coords[0], hp_patch_coords[1]]
        patch_confidence = self.__confidence[hp_pixel[0], hp_pixel[1]]

        for point in update_positions:
            self.__confidence[point[0], point[1]] = patch_confidence

        mask = self.__patch_data(self.__mask, hp_patch_coords)
        rgb_mask = self._to_rgb(mask)
        old_img_patch = self.__patch_data(self.__image, hp_patch_coords)
        new_img_patch = self.__patch_data(self.__image, new_patch)

        new = new_img_patch * rgb_mask + old_img_patch * (1 - rgb_mask)

        self.__latch(self.__image, hp_patch_coords, new)
        self.__latch(self.__mask, hp_patch_coords, 0)


    def __latch(self, image, path_coords, path_data):
        x1, x2, y1, y2 = path_coords
        image[x1:(x2 + 1), y1:(y2 + 1)] = path_data

    def __best_patch(self, center: tuple):
        patch_coords = self.__patch_coords_by_center(center)
        patch_height = patch_coords[3] - patch_coords[2] + 1
        patch_width = patch_coords[1] - patch_coords[0] + 1

        best_patch = None
        best_diff = np.inf

        lab_img = rgb2lab(self.__image)

        for y in range(self.__height - patch_height + 1):
            for x in range(self.__width - patch_width + 1):
                current_patch_coords = (x, x + patch_width - 1, y, y + patch_height - 1)
                
                if self.__patch_data(self.__mask, current_patch_coords).sum() != 0:
                    continue

                diff = self.patch_error(lab_img, current_patch_coords, patch_coords)

                if diff < best_diff:
                    best_diff = diff
                    best_patch = current_patch_coords

        return best_patch
    
    def patch_error(self, image, patch1_coords: tuple, patch2_coords: tuple):
        mask = 1 - self.__patch_data(self.__data, patch1_coords)
        rgb_mask = self._to_rgb(mask)

        patch1 = self.__patch_data(image, patch1_coords) * rgb_mask
        patch2 = self.__patch_data(image, patch2_coords) * rgb_mask

        color_distance = ((patch1 - patch2) ** 2).sum()
        coords_distance = np.sqrt((patch1_coords[0] - patch2_coords[0]) ** 2 + (patch1_coords[2] - patch2_coords[2]) ** 2)

        return color_distance + coords_distance

    def __update_front(self):
        self.__front = (laplace(self.__mask) > 0).astype(np.uint8)

    def __update_priority(self):
        self.__update_confidence()
        self.__update_data()

        self.__priority = self.__confidence * self.__data * self.__front

    def __update_data(self):
        normal = self.__normal()
        grad = self.__grad()

        data = normal * grad
        self.__data = np.sqrt(data[:, :, 0] ** 2 * data[:, :, 1] ** 2) + 0.001

    def __normal(self):
        x_sob = sobel(self.__mask, 0)
        y_sob = sobel(self.__mask, 1)
        normal = np.dstack((x_sob, y_sob))
        normalize = np.sqrt(y_sob**2 + x_sob**2).repeat(2).reshape(self.__mask.shape + (2,))
        normalize[normalize == 0] = 1
        return normal/normalize
    
    def __grad(self):
        grey_img = rgb2gray(self.__image)
        grey_img[self.__mask == 1] = None
        gradient = np.nan_to_num(np.gradient(grey_img))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros((*self.__mask.shape[:2], 2))
        
        front = np.argwhere(self.__front == 1)

        for point in front:
            patch = self.__patch_coords_by_center(point)
            patch_y_gradient = self.__patch_data(gradient[0], patch)
            patch_x_gradient = self.__patch_data(gradient[1], patch)
            patch_gradient_val = self.__patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]

        return max_gradient

    def __update_confidence(self):
        confidence = np.copy(self.__confidence)
        front_positions = np.argwhere(self.__front == 1)

        for x, y in front_positions:
            coords = self.__patch_coords_by_center((x, y))
            patch_area = self.__patch_area(coords)

            confidence[x, y] = np.sum(np.concatenate(self.__patch_data(self.__confidence, coords))) / patch_area

        self.__confidence = confidence

    def __check_full_coverage(self):
        return self.__mask.sum() == 0

    def apply(self, mask: np.ndarray, patch_size: int = 8):
        if mask.shape != self.__image.shape[:2]:
            raise ValueError("Mask shape must be the same as the image shape")

        self.__patch_size = patch_size
        self.__mask = np.copy(mask)
        self.__confidence = (1 - self.__mask).astype(np.float32)
        self.__data = np.zeros(self.__image.shape, dtype=np.float32)

        self.__inpaint()
        return np.copy(self.__image)

    def __highest_priority_patch_center(self):
        return np.unravel_index(self.__priority.argmax(), self.__priority.shape)

    def __patch_coords_by_center(self, point: tuple):
        half_patch_size = (self.__patch_size - 1) // 2
        height, width = self.__image.shape[:2]

        patch = [
            max(0, point[0] - half_patch_size),
            min(point[0] + half_patch_size, height - 1),
            max(0, point[1] - half_patch_size),
            min(point[1] + half_patch_size, width - 1)
        ]

        return patch
    
    def __patch_data(self, image, patch_coords: tuple):
        x1, x2, y1, y2 = patch_coords
        return image[x1:x2 + 1, y1:y2 + 1]
    
    def __patch_area(self, patch_coords: tuple):
        x1, x2, y1, y2 = patch_coords
        return (1 + x2 - x1) * (1 + y2 - y1)
    
    @staticmethod
    def _to_rgb(image):
        h, w = image.shape[:2]
        return image.reshape(h, w, 1).repeat(3, axis=2)
    
    @property
    def shape(self):
        return self.__image.shape

def main():
    if len(sys.argv) != 2:
        print("Usage: python inpaint.py <image>")
        sys.exit(1)

    img_path = sys.argv[1]

    inpainter = Inpainter(img_path)
    mask = np.zeros(inpainter.shape[:2], dtype=np.uint8)
    mask[0:25, 0:25] = 1

    Image.fromarray(mask * 255).show()

    inpainted = inpainter.apply(mask)
    Image.fromarray(inpainted).show()

if __name__=="__main__":
    main()
