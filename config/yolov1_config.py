from easydict import EasyDict
import os

__C = EasyDict()
cfg = __C


__C.DATA_PATH = 'data'
__C.PASCAL_PATH = os.path.join('', __C.DATA_PATH)
