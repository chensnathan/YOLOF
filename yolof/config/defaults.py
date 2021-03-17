from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER.BACKBONE_MULTIPLIER = 0.334
_C.SOLVER.WEIGHT_DECAY_NORM = 0.

# -----------------------------------------------------------------------------
# JitterCrop Transformation
# -----------------------------------------------------------------------------
_C.INPUT.JITTER_CROP = CN()
_C.INPUT.JITTER_CROP.ENABLED = False
_C.INPUT.JITTER_CROP.JITTER_RATIO = 0.3

# -----------------------------------------------------------------------------
# Resize Transformation
# -----------------------------------------------------------------------------
_C.INPUT.RESIZE = CN()
_C.INPUT.RESIZE.ENABLED = False
_C.INPUT.RESIZE.SHAPE = (640, 640)
_C.INPUT.RESIZE.SCALE_JITTER = (0.8, 1.2)
_C.INPUT.RESIZE.TEST_SHAPE = (608, 608)

# -----------------------------------------------------------------------------
# Distortion Transformation
# -----------------------------------------------------------------------------
_C.INPUT.DISTORTION = CN()
_C.INPUT.DISTORTION.ENABLED = False
_C.INPUT.DISTORTION.HUE = 0.1
_C.INPUT.DISTORTION.SATURATION = 1.5
_C.INPUT.DISTORTION.EXPOSURE = 1.5

# -----------------------------------------------------------------------------
# Shift Transformation
# -----------------------------------------------------------------------------
_C.INPUT.SHIFT = CN()
_C.INPUT.SHIFT.SHIFT_PIXELS = 32

# -----------------------------------------------------------------------------
# Mosaic Transformation
# -----------------------------------------------------------------------------
_C.INPUT.MOSAIC = CN()
_C.INPUT.MOSAIC.ENABLED = False
_C.INPUT.MOSAIC.POOL_CAPACITY = 1000
_C.INPUT.MOSAIC.NUM_IMAGES = 4
_C.INPUT.MOSAIC.MIN_OFFSET = 0.2
_C.INPUT.MOSAIC.MOSAIC_WIDTH = 640
_C.INPUT.MOSAIC.MOSAIC_HEIGHT = 640

# -----------------------------------------------------------------------------
# Anchor generator options
# -----------------------------------------------------------------------------
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
_C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

# -----------------------------------------------------------------------------
# BACKBONE
# -----------------------------------------------------------------------------
# DarkNet
_C.MODEL.DARKNET = CN()
_C.MODEL.DARKNET.DEPTH = 53
_C.MODEL.DARKNET.WITH_CSP = True
_C.MODEL.DARKNET.OUT_FEATURES = ["res5"]
_C.MODEL.DARKNET.NORM = "BN"
_C.MODEL.DARKNET.RES5_DILATION = 1

# -----------------------------------------------------------------------------
# YOLOF
# -----------------------------------------------------------------------------
_C.MODEL.YOLOF = CN()

# YOLOF Encoder parameters
# Note that the list of dilations must be consistent with number of blocks
_C.MODEL.YOLOF.ENCODER = CN()
_C.MODEL.YOLOF.ENCODER.BACKBONE_LEVEL = "res5"
_C.MODEL.YOLOF.ENCODER.IN_CHANNELS = 2048
_C.MODEL.YOLOF.ENCODER.NUM_CHANNELS = 512
_C.MODEL.YOLOF.ENCODER.BLOCK_MID_CHANNELS = 128
_C.MODEL.YOLOF.ENCODER.NUM_RESIDUAL_BLOCKS = 4
_C.MODEL.YOLOF.ENCODER.BLOCK_DILATIONS = [2, 4, 6, 8]
_C.MODEL.YOLOF.ENCODER.NORM = "BN"
_C.MODEL.YOLOF.ENCODER.ACTIVATION = "ReLU"

# YOLOF Decoder parameters
_C.MODEL.YOLOF.DECODER = CN()
_C.MODEL.YOLOF.DECODER.IN_CHANNELS = 512
_C.MODEL.YOLOF.DECODER.NUM_CLASSES = 80
_C.MODEL.YOLOF.DECODER.NUM_ANCHORS = 5
_C.MODEL.YOLOF.DECODER.CLS_NUM_CONVS = 2
_C.MODEL.YOLOF.DECODER.REG_NUM_CONVS = 4
_C.MODEL.YOLOF.DECODER.NORM = "BN"
_C.MODEL.YOLOF.DECODER.ACTIVATION = "ReLU"
_C.MODEL.YOLOF.DECODER.PRIOR_PROB = 0.01

# YOLOF box2box transform
_C.MODEL.YOLOF.BOX_TRANSFORM = CN()
_C.MODEL.YOLOF.BOX_TRANSFORM.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
_C.MODEL.YOLOF.BOX_TRANSFORM.ADD_CTR_CLAMP = True
_C.MODEL.YOLOF.BOX_TRANSFORM.CTR_CLAMP = 32

# YOLOF Uniform Matcher
_C.MODEL.YOLOF.MATCHER = CN()
_C.MODEL.YOLOF.MATCHER.TOPK = 4
# YOLOF ignore thresholds
_C.MODEL.YOLOF.POS_IGNORE_THRESHOLD = 0.15
_C.MODEL.YOLOF.NEG_IGNORE_THRESHOLD = 0.7

# YOLOF losses
_C.MODEL.YOLOF.LOSSES = CN()
_C.MODEL.YOLOF.LOSSES.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.YOLOF.LOSSES.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.YOLOF.LOSSES.BBOX_REG_LOSS_TYPE = "giou"

# YOLOF test
_C.MODEL.YOLOF.SCORE_THRESH_TEST = 0.05
_C.MODEL.YOLOF.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.YOLOF.NMS_THRESH_TEST = 0.6
_C.MODEL.YOLOF.DETECTIONS_PER_IMAGE = 100
