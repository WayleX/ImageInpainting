
def main():
    print('!')

class Inpainter:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def _load_photo(self, path_to_photo):
        pass
    def _load_mask(self, path_to_mask):
        pass
    def _validation(self):
        pass
    def inpaint(self, path_to_photo, path_to_mask, path_to_result):
        self._load_photo(path_to_photo)
        self._load_mask(path_to_mask)
        self._validation()
        
    

if __name__=="__main__":
    main()