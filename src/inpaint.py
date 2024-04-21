import sys
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
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
        while self.__mask.sum() != 0:
            self.__update_front()
            self.__update_priority()
            
            hp_patch_center = self.__highest_priority_patch_center()
            new_patch = self.__best_patch(hp_patch_center)

            self.__update(hp_patch_center, new_patch)
            print(f"Inpainting progress: {(1 - self.__mask.sum() / self.__mask.size) * 100:0.3f}%", end="\r")

    def __update(self, hp_pixel, new_patch):
        hp_patch_coords = self.__patch_coords_by_center(hp_pixel)

        update_positions = np.argwhere(
            self.__patch_data(self.__mask, hp_patch_coords) == 1) + [hp_patch_coords[0][0], hp_patch_coords[1][0]]
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
        image[
            path_coords[0][0]:path_coords[0][1]+1,
            path_coords[1][0]:path_coords[1][1]+1
        ] = path_data

    def __best_patch(self, hp_pixel: tuple):
        hp_patch = self.__patch_coords_by_center(hp_pixel)
        height, width = self.__image.shape[:2]
        patch_height, patch_width = (1 + hp_patch[0][1] - hp_patch[0][0]), (1 + hp_patch[1][1] - hp_patch[1][0])

        best_match = None
        best_match_difference = np.inf

        lab_image = rgb2lab(self.__image)

        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self.__patch_data(self.__mask, patch).sum() != 0:
                    continue

                difference = self.__patch_error(
                    lab_image,
                    hp_patch,
                    patch
                )

                if difference < best_match_difference:
                    best_match = patch
                    best_match_difference = difference

        return best_match

    def __patch_error(self, image, patch1_coords: tuple, patch2_coords: tuple):
        mask = 1 - self.__patch_data(self.__mask, patch1_coords)
        rgb_mask = self._to_rgb(mask)

        patch1 = self.__patch_data(image, patch1_coords) * rgb_mask
        patch2 = self.__patch_data(image, patch2_coords) * rgb_mask

        color_distance = ((patch1 - patch2) ** 2).sum()
        coords_distance = self.__patch_distance(patch1_coords, patch2_coords)

        return color_distance + coords_distance
    
    def __patch_distance(self, patch1_coords: tuple, patch2_coords: tuple):
        return np.sqrt(
            (patch1_coords[0][0] - patch2_coords[0][0]) ** 2 +
            (patch1_coords[1][0] - patch2_coords[1][0]) ** 2
        )

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
        self.__data = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2) + 0.001

    def __normal(self):
        x_sob = sobel(self.__mask, 0)
        y_sob = sobel(self.__mask, 1)

        normal = np.dstack((x_sob, y_sob))
        normalize = np.sqrt(y_sob ** 2 + x_sob ** 2).repeat(2).reshape(self.__height, self.__width, 2)
        normalize[normalize == 0] = 1
        
        return normal / normalize
    
    def __grad(self):
        grey_img = rgb2gray(self.__image)
        grey_img[self.__mask == 1] = None
        gradient = np.nan_to_num(np.gradient(grey_img))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros((self.__height, self.__width, 2), dtype=np.float32)

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
            patch_area = (1 + coords[0][1] - coords[0][0]) * (1 + coords[1][1] - coords[1][0])
            confidence[x, y] = sum(sum(self.__patch_data(self.__confidence, coords))) / patch_area

        self.__confidence = confidence

    def apply(self, mask: np.ndarray, patch_size: int = 9):
        if mask.shape != self.__image.shape[:2]:
            raise ValueError("Mask shape must be the same as the image shape")

        self.__patch_size = patch_size
        self.__mask = np.copy(mask)
        self.__confidence = (1 - self.__mask).astype(np.float32)
        self.__data = np.zeros(self.__image.shape[2:], dtype=np.float32)

        self.__inpaint()
        return np.copy(self.__image)

    def __highest_priority_patch_center(self):
        return np.unravel_index(self.__priority.argmax(), self.__priority.shape)

    def __patch_coords_by_center(self, point: tuple):
        half_patch_size = (self.__patch_size - 1) // 2
        height, width = self.__image.shape[:2]

        patch = [ 
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height-1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width-1)
            ]
        ]

        return patch
    
    @staticmethod
    def __patch_data(src, patch_coords):
        return src[
            patch_coords[0][0]:patch_coords[0][1] + 1,
            patch_coords[1][0]:patch_coords[1][1] + 1
        ]
    
    @staticmethod
    def _to_rgb(image):
        h, w = image.shape[:2]
        return image.reshape(h, w, 1).repeat(3, axis=2)
    
    @property
    def shape(self):
        return self.__image.shape

def main():
    if len(sys.argv) != 3:
        print("Usage: python inpaint.py <image> <mask>")
        sys.exit(1)

    img_path = sys.argv[1]
    mask_path = sys.argv[2]

    inpainter = Inpainter(img_path)
    mask = np.array(Image.open(mask_path))
    mask = np.floor(np.sum(mask, axis=2) // 765).astype(np.uint8)

    result = inpainter.apply(mask)
    Image.fromarray(result).show()

if __name__=="__main__":
    main()
