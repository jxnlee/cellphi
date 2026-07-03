from cellpose import core, models, io, utils
import numpy as np
import cv2
from skimage import measure
from scipy.spatial import distance
from tqdm import tqdm

class Cellphi:
    """ Class that represents a Cellphi configuration. The basis from which to perform rescaling, segmentation, and measurements.
    Attributes:
     - gpu_enabled (bool): indicates whether to run segmentation on GPU. True for GPU, False for only CPU.
     - scale_factor (float): number of real measurement units for each pixel
     - distance_unit (str): desired unit of measurement (eg. "um", "cm", etc.)
     - target_width (int): target horizontal resolution for output
     - target_height (int): target vertical resolution for output
    Methods:
    - enable_gpu(self): if GPU is supported, then attempts to enable it for segmentation
    - disable_gpu(self): disables using GPU in segmentation
    - set_resolution(self, width=0, height=0, reference=None): sets target resolution based either on 
        a specified number of pixels or a reference image
    """

    def __init__(self, scale_factor=1.0, unit="um", width=0, height=0, reference=None, enable_gpu=False):
        if enable_gpu:
            self.enable_gpu()
        else:
            self.disable_gpu()
        self.set_resolution(width=width, height=height, reference=reference)
        self.scale_factor = scale_factor
        self.unit = unit

    def resize(self, img):
        try:
            return cv2.resize(img, (self.target_width, self.target_height))
        except Exception as e:
            print(f"ERROR Resizing Image: {e}")
    
    def pose(self, img=None, imgs=None, burst=False, enable_logger=True, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, max_size_fraction=0.4):
        if enable_logger:
            io.logger_setup()
        print("-> Running Cellpose Segmentation...")
        if not burst and img is not None:
            masks, _, _ = self.model.eval(img)
        elif burst and imgs is not None:
            masks, _, _ = self.model.eval(imgs)
        else:
            raise ValueError(f"Need singular image without burst or list of images with burst.")
        return masks
        
    def take(self, img, masks, measurements=['max_feret_diameter'], draw=['outline', 'tag', 'max_feret_diameter']):
        taken_img = img.copy()
        cells = self._cells_from_masks(masks)
        # outlines = utils.outlines_list(masks)

        record_data = False
        data = []
        if 'tag' in draw:
            record_data = True
        
        if taken_img.dtype != np.uint8:
            taken_img = (taken_img * 255).astype(np.uint8)

        print(f"-> Found {len(cells)} cells. Taking measurements...")

        for idx, (cell_id, outline) in enumerate(tqdm(cells, desc="Measuring")):
            if len(outline) < 2:
                continue

            measured = {}

            if 'outline' in draw:
                pts = outline.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(taken_img, [pts], isClosed=True, color=(0, 255, 255), thickness=1)
            # START OF MEASUREMENTS
            if 'max_feret_diameter' in measurements:
                measured['max_feret_diameter'] = self._max_feret_diameter(taken_img, outline, draw='max_feret_diameter' in draw)
            # END OF MEASUREMENTs
            if record_data:
                data.append(measured)
            
            if 'tag' in draw:
                self._text(taken_img, outline, str(idx))
            elif 'label' in draw:
                self._text(taken_img, outline, f"{measured['max_feret_diameter']:.3f}")
        return taken_img, data
    
    def _cells_from_masks(self, masks):
        cells = []
        labeled_masks = measure.label(masks)
        for cell_id in np.unique(labeled_masks):
            if cell_id == 0:
                continue

            cell_mask = (labeled_masks == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            outline = max(contours, key=cv2.contourArea).reshape(-1, 2)
            cells.append((cell_id, outline))
        return cells

    def _text(self, img, outline, text, color=(0, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.2, thickness=1):
        x = int(np.mean(outline[:, 0]))
        y = int(np.mean(outline[:, 1]))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x - text_width // 2
        text_y = y + text_height // 2
        cv2.rectangle(img, (text_x - 2, text_y - text_height - 2),
                      (text_x + text_width + 2, text_y + baseline),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _scale(self, pixels):
        return pixels * self.scale_factor
    
    def _max_feret_diameter(self, img, outline, draw=True, color=(0, 255, 255), thickness=1):
        full_dist_matrix = distance.cdist(outline, outline)
        max_dist_px = np.max(full_dist_matrix)
        max_dist_units = self._scale(max_dist_px)

        max_dist_indices = np.unravel_index(np.argmax(full_dist_matrix), full_dist_matrix.shape)
        p1 = outline[max_dist_indices[0]]
        p2 = outline[max_dist_indices[1]]

        if draw:
            pt1_int = tuple(p1.astype(np.int32))
            pt2_int = tuple(p2.astype(np.int32))
            cv2.line(img, pt1_int, pt2_int, color=color, thickness=thickness)

        return max_dist_units
    
    def enable_gpu(self):
        self.gpu_enabled = core.use_gpu()
        if not self.gpu_enabled:
            print("WARNING: GPU requested but not available. Falling back to CPU.")
        print("-> Loading Cellpose Model...")
        self.model = models.CellposeModel(gpu=self.gpu_enabled)
    
    def disable_gpu(self):
        print("WARNING: GPU not enabled. Running on CPU. Expect extended processing time during segmentation.")
        self.gpu_enabled = False
        print("-> Loading Cellpose Model...")
        self.model = models.CellposeModel(gpu=self.gpu_enabled)

    def set_resolution(self, width=0, height=0, reference=None):
        try:
            if reference is not None:
                if isinstance(reference, str):
                    ref_img = cv2.imread(reference)
                    if ref_img is None:
                        raise FileNotFoundError(f"Reference image not found at {reference}")
                    self.target_height, self.target_width, _ = ref_img.shape
                elif isinstance(reference, np.ndarray):
                    self.target_height, self.target_width, _ = reference.shape
                else:
                    raise ValueError("Reference must be a file path or a numpy array image format.")
            else:
                if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                    self.target_width = width
                    self.target_height = height
                else:
                    raise ValueError("Width and Height must be specified for target resolution.")
        except Exception as e:
            raise ValueError(f"Error setting resolution: {e}.")
    