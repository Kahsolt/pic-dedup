#!/usr/bin/env python3

import logging
import pickle
import numpy as np
from enum import IntEnum
from PIL import Image, ImageFilter, ImageDraw, ImageFont

from .settings import *
from .models import *

__all__ = ['SimRatioMatrix', 'Feature', 'pkl', 'HWRatio', 'resize_by_hwlimit', 'high_contrast_bw_hexstr']

class SimRatioMatrix:
  
  TYPES_DEPTH = 3  # 3 types of sim_ratio: edge_avghash, grey_avghash, grey_absdiff
  MASK_DEPTH = 2   # whether use round_mask
  
  def __init__(self, size):
    self.size = size
    self.sr_mat = np.full((size, size, self.TYPES_DEPTH, self.MASK_DEPTH), 0.0, dtype=np.float32)
    self.modified = False       # indicates pkl('save')
  
  def __setitem__(self, xy, val):
    try:
      self.sr_mat[xy] = val
    except IndexError:
      self.expand(max(xy))
      self.sr_mat[xy] = val
    self.modified = True
    
  def __getitem__(self, xy):
    try:
      return self.sr_mat[xy]
    except IndexError:
      self.expand(max(xy))
      self.modified = True
      return self.sr_mat[xy]
  
  def expand(self, newsize):
    if newsize <= self.size: return
    logging.debug("[%s] expand from %dx%d to %dx%d" 
                  % (self.__class__.__name__, self.size, self.size, newsize, newsize))
    
    sr_mat = np.full((newsize, newsize, self.TYPES_DEPTH, self.MASK_DEPTH), 0.0, dtype=np.float32)
    _sz = self.size
    sr_mat[0:_sz, 0:_sz] = self.sr_mat[0:_sz, 0:_sz]
    self.sr_mat = sr_mat
    self.size = newsize
  
  @staticmethod
  def from_bytes(bytes):
    return pickle.loads(bytes)

  def to_bytes(self):
    return pickle.dumps(self)

class PrincipleHues:
  
  def __init__(self, phs):
    if not isinstance(phs, list): raise TypeError

    self.phs = phs    # list of 3-tuples [(R, G, B)]
    self.phs_hexstr = [rgb2hexstr(ph) for ph in phs]
    
  @staticmethod
  def from_image(img, count):
    # thumbnailize
    _hw = FEATURE_VECTOR_HW
    img = resize_by_hwlimit(img, _hw)

    # mosaic filter
    #  keep 0 as unchanged (H = S = 0 means pure greyness in HSV space)
    #  offset by 0.5 for linear interplotion
    img = img.convert('HSV')
    hsv = list(img.split())
    for i in range(len(hsv)):
      _ratio = 256 // REDUCED_HUE_SCALES[i]
      hsv[i] = hsv[i].point(lambda x: x and int((x // _ratio + 0.5) * _ratio) or 0)
    img = Image.merge('HSV', hsv)

    # mode filter to reduce hues
    img = img.filter(ImageFilter.ModeFilter((_hw >> 3) + 1))
    img = img.convert('RGB')

    # decide priciple ones
    phs = [ ]
    for hue in [rgb for _, rgb in sorted(img.getcolors(_hw ** 2), reverse=True)]:
      ignore = False
      for ph in phs:
        if rgb_distance(hue, ph) < HUE_DISTINGUISH_DISTANCE:
          ignore = True
          break
      if ignore: continue
      phs.append(hue)
      if len(phs) == count: break
    
    return PrincipleHues(phs)
  
  def to_image(self):
    _hw = 50
    img = Image.new('RGB', (_hw, _hw * len(self.phs)))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('arial.ttf')
    for i, ph in enumerate(self.phs):
      xy = (0, i * _hw), (_hw, (i + 1) * _hw)
      draw.rectangle(xy, fill=ph)
      xy = (0, i * _hw)
      draw.text(xy, self.phs_hexstr[i], high_contrast_bw_hexstr(ph), font=font)
    return img
  
  def compability(self, hue):
    mindist = HUE_MAX_DISTANCE + 1
    _alpha, _portion = 1.0, 0.6 / len(self.phs)
    for ph in self.phs:
      dist = rgb_distance(ph, hue)
      if dist < mindist:
        mindist, alpha = dist, _alpha
      _alpha -= _portion
    return (1 - (mindist / HUE_MAX_DISTANCE)) * alpha
    
class FeatureVector:

  def __init__(self, featvec):
    if not isinstance(featvec, np.ndarray): raise TypeError
    if len(featvec.shape) != 2: raise ValueError

    self.fv = featvec
    self.fv_len = np.prod(self.fv.shape)  # flattened length
    _mean = np.uint8(self.fv[self.fv != NONE_PIXEL_PADDING].mean())
    self.fv_bin = np.array([  # FIXME: x <= mean is risky for single color image
        [0 if x == NONE_PIXEL_PADDING else (x <= _mean and 1 or -1)]
        for row in self.fv for x in row], dtype=np.int8)
    self.fv_masked = None     # calc and save on necessaryd

  @staticmethod
  def from_image(img, hw):
    imgpad = square_padding(img)
    thumbnail = imgpad.resize((hw, hw), Image.ANTIALIAS)
    im = np.array(thumbnail,  dtype=np.uint8)
    return FeatureVector(im)

  def to_image(self):
    return Image.fromarray(self.fv).convert('L')

  def similarity_by_avghash(self, other):
    if self is other: return 1.0
  
    dist = 0
    for idx, row in enumerate(self.fv_bin):
      for idy, x in enumerate(row):
        y = other.fv_bin[idx, idy]
        # according to this table:
        #    x^y -1  0  1
        #    -1   0  1 -2
        #     0   1  0  1
        #     1  -2  1  0
        if x ^ y == -2:         # one 1 and one -1
          dist += 2
        elif abs(x ^ y) == 1:   # one 0 and one other
          dist += 1
    return 1 - (dist / 2 / self.fv_len)

  def similarity_by_absdiff(self, other):
    if self is other: return 1.0

    dist = 0.0
    for idx, row in enumerate(self.fv):
      for idy, x in enumerate(row):
        y = other.fv[idx, idy]
        if (x != NONE_PIXEL_PADDING) and (y != NONE_PIXEL_PADDING):
          dist += abs(int(x) - int(y))
        elif (x == NONE_PIXEL_PADDING) ^ (y == NONE_PIXEL_PADDING):
          dist += 127.0
    return 1 - (dist / 255 / self.fv_len)
  
  def round_mask(self):
    mfv = self.fv_masked
    if mfv is None:
      r = self.fv.shape[0] / 2
      mfv = np.full_like(self.fv, NONE_PIXEL_PADDING, dtype=np.uint8)
      for idx, row in enumerate(self.fv):
        for idy, x in enumerate(row):
          if (r - idx) ** 2 + (r - idy) ** 2 <= r ** 2:
            mfv[idx, idy] = x
      self.fv_masked = mfv

    return FeatureVector(mfv)

class Feature:
  
  def __init__(self):
    self.principle_hues = None  # instance of PrincipleHues
    self.featvec_edge = None    # instance of  FeatureVector
    self.featvec_grey = None

  @staticmethod
  def featurize(img, hw=FEATURE_VECTOR_HW, phs=PRINCIPLE_HUES):
    if isinstance(img, Image.Image): pass
    elif isinstance(img, np.ndarray): img = Image.fromarray(img)
    elif isinstance(img, str): img = Image.open(img)
    else: raise TypeError

    ft = Feature()
    img = img.convert('RGB')
    ft.principle_hues = PrincipleHues.from_image(img, phs)
    grey = img.convert('L')
    ft.featvec_grey = FeatureVector.from_image(grey, hw)
    ft.featvec_grey._parent = ft  # backref of Feature
    edge = grey.filter(ImageFilter.CONTOUR)   # .filter(ImageFilter.EDGE_ENHANCE_MORE)
    ft.featvec_edge = FeatureVector.from_image(edge, hw)
    ft.featvec_edge._parent = ft  # backref of Feature
    return ft
  
  @staticmethod
  def from_bytes(bytes):
    return pickle.loads(bytes)

  def to_bytes(self):
    return pickle.dumps(self)

def pkl(what='load', model=None):
  # auxiliary onvert function for pickled data in models
  if not model: return

  if isinstance(model, Folder):
    fld = model
    if what == 'load':
      sr_mat = fld.sr_matrix_pkl
      if not sr_mat:
        with db_lock: sz = int(fld.pictures.count() * 1.5)
        sr_mat = SimRatioMatrix(sz).to_bytes()
        fld.sr_matrix_pkl = sr_mat
        save(fld)
      fld.sr_matrix = SimRatioMatrix.from_bytes(sr_mat)

    elif what == 'save':
      fld.sr_matrix_pkl = fld.sr_matrix.to_bytes()
      save(fld)

  elif isinstance(model, Picture):
    pic = model
    if what == 'load':
      ft = pic.feature_pkl
      if not ft:
        ft = Feature.featurize(pic.path)
        pic.feature_pkl = ft
        save(pic)
      pic.feature = Feature.from_bytes(ft)

class HWRatio(IntEnum):
  # item named <shape>_<width>_<height>, but value is hwr = height / width
  SQUARE_1_1 =      100
  HORIZONTAL_4_3 =  100 * 3 // 4
  HORIZONTAL_3_2 =  100 * 2 // 3
  HORIZONTAL_16_9 = 100 * 9 // 16
  VERTICLE_3_4 =    100 * 4 // 3
  VERTICLE_2_3 =    100 * 3 // 2
  VERTICLE_9_16 =   100 * 16 // 9
  
def resize_by_hwlimit(img, hwlimit=640, sample=Image.NEAREST):
  # shrink image by given hwlimit, with aspect ratio kept
  # use default NEAREST keeps original color other than interplotion
  if max(img.size) > hwlimit:
    w, h = img.size
    if w >= h:
      sz = (hwlimit, h * hwlimit // w)
    else:
      sz = (w * hwlimit // h, hwlimit)
    img = img.resize(sz, sample)
  return img

def square_padding(img):
  # expand to fit a minimal square canvas
  w, h = img.size
  if w == h: return img

  # let's use the magic number 255 to represent the
  # padded invalid pixels, so adjust all REAL 255 to 254
  im = np.array(img, dtype=np.uint8)
  im[im == NONE_PIXEL_PADDING] = NONE_PIXEL_PADDING - 1

  mhw = max(img.size)
  impad = np.full((mhw, mhw), NONE_PIXEL_PADDING, dtype=np.uint8)
  if w > h:
    _len = (mhw * h) // w
    _y = (mhw - _len) >> 1
    rs, re = _y, _y + _len
    cs, ce = 0, w
  else:
    _len = (mhw * w) // h
    _x = (mhw - _len) >> 1
    rs, re = 0, h
    cs, ce = _x, _x + _len
  impad[rs:re, cs:ce] = im[0:h, 0:w]

  return Image.fromarray(impad).convert('L')

def rgb2grey(rgb) -> int:
  # ITU-R 601-2 luma transform
  r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
  return int((r * 299 + g * 587 + b * 114) // 1000)

def rgb2hexstr(rgb) -> str:
  r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
  return '#' + hex((r << 16) + (g << 8) + (b))[2:].rjust(6, '0')

def rgb_distance(rgb1, rgb2) -> float:
  # distance in LAB color space, but input is RGB
  # see: https://blog.csdn.net/qq_16564093/article/details/80698479
  R, G, B = [x - y for x, y in zip(rgb1, rgb2)]
  rmean = (rgb1[0] + rgb2[0]) / 2
  c = np.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))
  return float(c)

def high_contrast_bw_hexstr(rgb):
  return rgb2grey(rgb) <= 192 and '#FFFFFF' or '#000000'


def hsv2rgb(hsv) -> (int, int, int):
  h, s, v = float(hsv[0] / 255.0 * 360), float(hsv[1] / 255.0), float(hsv[2] / 255.0)
  h60 = h / 60.0
  h60f = np.floor(h60)
  hi = int(h60f) % 6
  f = h60 - h60f
  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)

  r, g, b = 0, 0, 0
  if hi == 0: r, g, b = v, t, p
  elif hi == 1: r, g, b = q, v, p
  elif hi == 2: r, g, b = p, v, t
  elif hi == 3: r, g, b = p, q, v
  elif hi == 4: r, g, b = t, p, v
  elif hi == 5: r, g, b = v, p, q

  return round(r * 255), round(g * 255), round(b * 255)

def rgb2hsv(rgb) -> (int, int, int):
  r, g, b = float(rgb[0] / 255.0), float(rgb[1] / 255.0), float(rgb[2] / 255.0)
  mx, mn = max(r, g, b), min(r, g, b)
  df = mx - mn

  if mx == mn: h = 0
  elif mx == r:
    h = (60 * ((g - b) / df) + 360) % 360
  elif mx == g:
    h = (60 * ((b - r) / df) + 120) % 360
  elif mx == b:
    h = (60 * ((r - g) / df) + 240) % 360
  s = 0 if mx == 0 else df / mx
  v = mx
  
  return round(h / 360 * 255), round(s * 255), round(v * 255)
