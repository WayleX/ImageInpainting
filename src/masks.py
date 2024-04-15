import numpy as np
import cv2

def square_patches_mask(size, num_patches, patch_size):
    mask = np.zeros(size, dtype=np.uint8)
    patches = []
    for _ in range(num_patches):
        placed = False
        while not placed:
            x = np.random.randint(0, size[1] - patch_size)
            y = np.random.randint(0, size[0] - patch_size)
            patch = [x, y, x+patch_size, y+patch_size]
            overlap = False
            for existing_patch in patches:
                if (patch[0] < existing_patch[2] and patch[2] > existing_patch[0] and
                    patch[1] < existing_patch[3] and patch[3] > existing_patch[1]):
                    overlap = True
                    break
            if not overlap:
                patches.append(patch)
                mask[y:y+patch_size, x:x+patch_size] = 255
                placed = True
    return mask

def random_noise_mask(size, density):
    mask = np.zeros(size, dtype=np.uint8)
    num_pixels = int(size[0] * size[1] * density)
    indices = np.random.choice(size[0] * size[1], num_pixels, replace=False)
    y_indices, x_indices = np.unravel_index(indices, size)
    mask[y_indices, x_indices] = 255
    return mask

def line_damage_mask(size, num_lines, line_width):
    mask = np.zeros(size, dtype=np.uint8)
    for _ in range(num_lines):
        x1 = np.random.randint(0, size[1])
        y1 = np.random.randint(0, size[0])
        x2 = np.random.randint(0, size[1])
        y2 = np.random.randint(0, size[0])
        cv2.line(mask, (x1, y1), (x2, y2), (255), thickness=line_width)
    return mask

if __name__ == '__main__':
    # Parameters
    size = (300, 300)
    num_patches = 10
    patch_size = 50
    density = 0.1
    num_lines = 7
    line_width = 2

    # Generating masks
    square_mask = square_patches_mask(size, num_patches, patch_size)
    noise_mask = random_noise_mask(size, density)
    line_mask = line_damage_mask(size, num_lines, line_width)

    # Displaying masks
    cv2.imshow('Square Patches Mask', square_mask)
    cv2.imshow('Random Noise Mask', noise_mask)
    cv2.imshow('Line Damage Mask', line_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
