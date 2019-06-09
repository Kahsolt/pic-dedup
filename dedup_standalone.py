#!/usr/bin/env python3
# This file is auto-generated, manual changes should be lost.
# build date: 2019-06-09 15:23:41.902364.

__dist__ = "standalone"

from PIL import Image
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from PIL import Image, ImageTk
from collections import defaultdict
from enum import IntEnum
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, relationship
from win32com.shell import shell, shellcon
import logging
import numpy as np
import os
import pickle
import queue
import random
import sqlalchemy as sql
import threading
import time
import tkinter as tk
import tkinter.colorchooser
import tkinter.filedialog
import tkinter.messagebox as tkmsg
import tkinter.simpledialog
import tkinter.ttk as ttk
import win32clipboard as cpd
import win32con

# __init__.py
__version__ = '0.1'

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(PACKAGE_PATH)

# settings.py
## 0. User Settings

# sim_thresh for simgrp empirical references:
#  0.70 ~ 0.80: pictures that are alike in colormaps or histograms
#  0.80 ~ 0.95: similar in hues and color-blocks, typically a picture series
#  0.95 ~ 1.00: proximately the same picture but only differs in resolution or size
# suggested value: 0.70 ~ 0.95
#
# sim_thresh for filter empirical references:
#  0.80 ~ 0.95: similiar hue presented
#  0.95 ~ 1.00: proximately the same hue presented
# suggested value: 0.85 ~ 0.95
SIMILAR_THRESHOLD = 0.80

# hw-ratio tolerance
# suggested value: 10 ~ 30
HWR_TOLERANCE = 20

# only compare the inscribed circle area
ROUND_MASK = False

# fast search config set:
#  1. only compare images within HWR_TOLERANCE
#  2. only compare edge featvec
FAST_SEARCH = True

# image feature vector width/height
# suggested value: 16, 32, 64
FEATURE_VECTOR_HW = 32

# principle hues count of a image
# suggested value: 3 ~ 5
PRINCIPLE_HUES = 4

# main window size
# suggested value: (800, 600), (912, 684), (1024, 768)
WINDOW_SIZE = (800, 600)

# picture hwlimit in preview, please set according to WINDOW_SIZE
# suggested value: 250, 300, 350
PREVIEW_HW = 250

# split preview
PREVIEW_SPLIT = True

## 1. System Settings

# index recursively into subfolders
RECURSIVE_INDEX = False

# image extionsions, indexer identify files by extension name
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']

# do not use these names as folder name, rather use its parents' basename
FOLDER_NAME_BLACKLIST = ['image', 'images', 'manga', 'mangas', 'picture', 'pictures', 'data', 'webdata']

# global sleep interval in seconds
# suggested value: 0.5 ~ 1.5
SLEEP_INTERVAL = 1.0

# worker count factor (e.g. set to 2 means 2 * os.cpu_count())
# suggested value: 2.0 ~ 4.0
WORKER_FACTOR = 3.0

# cache size of item count
# suggested value: 100 ~ 300
CACHE_SIZE = 150

# index database file, set None for auto configuring
DB_FILE = None

# auto reindex after reset db (this should take a century long..)
AUTO_REINDEX = False

## 2. System Constants (hard coded with careful design, DO NOT TOUCH !!)

# padding value of non-square images
# FIXED VALUE: 255 (pure white color pixel)
NONE_PIXEL_PADDING = 255

# reduce hues in HSV mode
# FIXED VALUE: (16, 8, 8)
REDUCED_HUE_SCALES = (16, 8, 8)

# reduce hues in HSV mode
# FIXED VALUE: 95 (damn empirical value)
HUE_DISTINGUISH_DISTANCE = 95

# const max retval of rgb_distance()
# FIXED VALUE: 764.8339663572415
HUE_MAX_DISTANCE = 764.8339663572415

# models.py
db = None
db_lock = threading.RLock()
Model = declarative_base()

class Folder(Model):

  @declared_attr
  def __tablename__(cls): return cls.__name__

  id = sql.Column(sql.INTEGER, primary_key=True, autoincrement=True)
  pictures = relationship('Picture', order_by='Picture.id', back_populates='folder', lazy='dynamic')

  path = sql.Column(sql.TEXT, unique=True, comment='absolute path')
  name = sql.Column(sql.TEXT, comment='folder basename or distinguishable human readable name')
  deleted = sql.Column(sql.BOOLEAN, default=False, comment='soft delete mark')

  sr_matrix_pkl = sql.Column(sql.BLOB, default=None, comment='pickled sim_ratio matrix')
  sr_matrix = None  # instance of np.ndarray, loaded from sr_matrix_pkl

  def __repr__(self):
    return '<%s id=%r name=%r path=%r deleted=%r>' % (
        self.__class__.__name__, self.id, self.name, self.path, self.deleted)

class Picture(Model):

  @declared_attr
  def __tablename__(cls): return cls.__name__

  id = sql.Column(sql.INTEGER, primary_key=True, autoincrement=True)
  folder_id = sql.Column(sql.ForeignKey('Folder.id', onupdate='CASCADE', ondelete='CASCADE'))
  folder = relationship('Folder', back_populates='pictures')

  path = sql.Column(sql.TEXT, comment='absolute path')
  filename = sql.Column(sql.TEXT, comment='file basename')
  width = sql.Column(sql.INT)
  height = sql.Column(sql.INT)
  hwr = sql.Column(sql.INT, comment='height / width * 100')
  size = sql.Column(sql.INT, comment='file size in bytes')
  deleted = sql.Column(sql.BOOLEAN, default=False, comment='soft delete mark, keeping rank in sr_matrix')

  feature_pkl = sql.Column(sql.BLOB, default=None, comment='pickled Feature')
  feature = None  # instance of Feature, loaded from feature_pkl

  def __repr__(self):
    return '<%s id=%r folder=%r path=%r width=%r height=%r, size=%r>' % (
        self.__class__.__name__, self.id, self.folder, self.path, self.width, self.height, self.size)

  def __lt__(self, other):
    return self.size < other.size

def detect_env():
  if globals().get('__dist__') == 'standalone':
    env = 'dist'
  else:
    env = 'dev'

  return env

def setup_db(dbname='index.db', env='dist'):
  global db
  if db: return   # fix pickle bug

  # configure database file
  dbfile = os.path.join(BASE_PATH, dbname)
  if DB_FILE:
    dbfile = DB_FILE
  elif env == 'dist':
    dbfile = os.path.join(PACKAGE_PATH, dbname)
  logging.info('[DB] use %s' % dbfile)

  # orm session
  engine = sql.create_engine('sqlite:///%s?check_same_thread=False' % dbfile)
  engine.execute('PRAGMA journal_mode = PERSIST;').fetchall()
  Model.metadata.create_all(engine)
  session_maker = sessionmaker(bind=engine)
  db = session_maker()

  # dev env auto setup hook
  if env == 'dev':
    logging.info('[DB] dev env setup & init...')
    dp = os.path.join(BASE_PATH, 'test')
    if db.query(Folder).filter_by(path=dp).count() == 0:
      fld = Folder()
      fld.name = os.path.basename(dp)
      fld.path = dp
      save(fld)

def save(model=None):
  with db_lock:
    if model: db.add(model)
    try: db.commit()
    except Exception as e: logging.error(e)

# global initialize
_env = detect_env()
logging.basicConfig(level=_env == 'dist' and logging.INFO or logging.DEBUG,
                    format="%(asctime)s - %(levelname)s: %(message)s")
setup_db(env=_env)

# imgproc.py
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

# utils.py
class LRUCache:

  class LRUCacheItem:
    data = None
    use = 0
    expire = True

    def __lt__(self, other):
      if self.expire == other.expire:
        return self.use <= other.use
      else:
        return self.expire and not other.expire

  def __init__(self, max_size=300):
    self.pool = { }
    self.max_size = max_size

    self.stat_hit = 0
    self.stat_miss = 0

    self.evt_stop = threading.Event()
    self.maintain_worker = threading.Thread(
        target=LRUCache.maintain_task, args=(self,),
        name='cache_cleaner')
    self.maintain_worker.start()

    logging.info('[%s] start pooling for at most %d items' % (self.__class__.__name__, max_size))

  def get(self, key):
    item = self.pool.get(key)
    if item:
      item.use += 1
      self.stat_hit += 1
      return item.data
    self.stat_miss += 1
    return None

  def set(self, key, val, expire=True):
    item = LRUCache.LRUCacheItem()
    item.data = val
    item.expire = expire
    self.pool[key] = item

  def clear(self):
    self.pool.clear()

  def destroy(self):
    self.evt_stop.set()
    self.pool = None
    self.maintain_worker.join()

    logging.debug('[%s] destroyed' % self.__class__.__name__)

  @staticmethod
  def maintain_task(cache):
    next_period = 10
    while not cache.evt_stop.is_set():
      while next_period:
        next_period -= 1
        time.sleep(SLEEP_INTERVAL)
        if cache.evt_stop.is_set(): return

      # stat
      hit, miss = cache.stat_hit, cache.stat_miss
      tot = hit + miss
      if tot: logging.debug('[%s] stat: %d hit, %d miss, hit_ratio = %.2f' %
                            (cache.__class__.__name__, hit, miss, hit / tot))

      # gc
      _ovf = len(cache.pool) - cache.max_size
      if _ovf > 0:
        logging.debug('[%s] %d items cleaned' % (cache.__class__.__name__, _ovf))
        for k in sorted(cache.pool)[:_ovf]:
          cache.data.pop(k)

      # approximately mapping: [0, 150+) => [60, 5]
      next_period = int(_ovf ** 2 / -400 + 60)
      if next_period < 5: next_period = 5

class UnionSet:

  def __init__(self, size):
    self.size = size
    self.father = np.full(size, -1, dtype=np.int32)

  def find(self, x):
    f = self.father[x]
    if f < 0: return x
    else:
      self.father[x] = self.find(f)
      return self.father[x]

  def in_same_set(self, x, y):
    return self.find(x) == self.find(y)

  def union(self, x, y):
    fx, fy = self.find(x), self.find(y)
    if fx != fy:
      if self.father[fx] < self.father[fy] or \
          self.father[fx] == self.father[fy] and fx < fy:
        self.father[fx] += self.father[fy]
        self.father[fy] = fx
      else:
        self.father[fy] += self.father[fx]
        self.father[fx] = fy

  def get_sets(self, min_size=1):
    grps = { }
    for x in range(self.size):
      f = self.find(x)
      if not grps.get(f):
        grps[f] = set()
      grps[f].add(x)

    return [grp for grp in grps.values() if len(grp) >= min_size]

class Scheduler:

  class DummyBulletin:

    def set(self, msg):
      logging.info('[Scheduler] %s' % msg)

  class DummyProgress:

    def start(self):
      logging.info('[Scheduler] working...')

    def stop(self):
      logging.info('[Scheduler] idle...')

  def __init__(self, workers=8):
    self.Q = queue.Queue()
    self.evt_stop = threading.Event()
    self.lock = threading.RLock()
    self.workers = [threading.Thread(
        target=self.__class__.worker, args=(self, i),
        name='worker-%d' % (i + 1))
      for i in range(workers)]
    for worker in self.workers: worker.start()
    self.idle = True
    self.workers_idle = [True] * workers

    self.bulletin = Scheduler.DummyBulletin()   # cound be reset later, inform use certain progress of tasks
    self.progress = Scheduler.DummyProgress()

    self.task_starttime = { }                   # { str(tag): int(T) }
    self.task_pending = defaultdict(lambda: 0)  # { str(tag): int(cnt) }
    self.task_finished = defaultdict(lambda: 0) # { str(tag_aggregated): int(cnt) }

    logging.info('[%s] starting with %d workers' % (self.__class__.__name__, workers))

  def set_stat_widgets(self, bulletin, progress_bar):
    self.bulletin = bulletin
    self.progress = progress_bar

  def stop(self):
    self.evt_stop.set()
    for worker in self.workers: worker.join()

    logging.debug('[%s] exited' % self.__class__.__name__)

  def add_task(self, tag, task, args=None, unique=False):
    '''
      tag: single-task name, or subtask name (starts with ':')
      task: callable which calls under args
    '''
    if self.idle: self.progress.start()

    if not tag: tag = '_'
    if tag in self.task_pending:
      if unique: return  # ignore on duplicate
    elif not tag.startswith(':'):
      self.task_starttime[tag] = time.time()
      self.bulletin.set("Task %r started.." % tag)

    self.task_pending[tag] += 1
    self.Q.put((tag, task, args))

  def update_progress(self, tag, total=None):
    rest = self.task_pending.get(tag)
    info = total and "Task %s finish %.2f%%..." % (tag, 100 * (total - rest) / total) \
                   or "Task %s pending %d..." % (tag, rest)
    logging.info("[%s] %s" % (self.__class__.__name__, info))
    self.bulletin.set(info)
    return rest == 0

  def _task_done(self, tag):
    with self.lock:
      self.task_pending[tag] -= 1
      if self.task_pending.get(tag) == 0:
        self.task_pending.pop(tag)
        if not tag.startswith(':'):
          T = time.time() - self.task_starttime.pop(tag)
          self.bulletin.set("Task %s done in %.2fs." % (tag, T))

      self.task_finished['-' in tag and tag.split("-")[0] or tag] += 1
      self.Q.task_done()

    if not self.Q.unfinished_tasks:
      self.progress.stop()
      self.idle = True

  def report(self):
    _nothing = 'Nothing :>'
    pending = '\n'.join([k + ': ' + str(self.task_pending[k]) for k in sorted(self.task_pending)])
    finished = '\n'.join([k + ': ' + str(self.task_finished[k]) for k in sorted(self.task_finished)])
    _cnt_idle = sum(self.workers_idle)
    workers = ('working: ' + str(len(self.workers) - _cnt_idle) +
               ', idle: ' + str(_cnt_idle))
    info = ("[Workers]\n" + workers +
            "\n\n[Pending]\n" + (pending or _nothing) +
            "\n\n[Done]\n" + (finished or _nothing))
    return info

  @staticmethod
  def worker(scheduler, i):
    logging.debug('[%s] %r started' % (scheduler.__class__.__name__, threading.current_thread().name))

    while not scheduler.evt_stop.is_set():
      tag, task, args = None, None, None
      while not task:
        try: tag, task, args = scheduler.Q.get(timeout=SLEEP_INTERVAL)
        except queue.Empty:
          if scheduler.evt_stop.is_set(): break
          time.sleep(0)   # yield
      if scheduler.evt_stop.is_set(): break

      # just do it!
      scheduler.workers_idle[i] = False
      if args: task(*args)  # the start * unpacks arguments
      else: task()
      scheduler._task_done(tag)
      scheduler.workers_idle[i] = True

    logging.debug('[%s] %r exited' % (scheduler.__class__.__name__, threading.current_thread().name))

def index(scheduler, tag, fid, dp,
          recursive=False):
  def _task(fid, fp):  # wrap to make a callable task
    def wrapper():
      img = Image.open(fp)
      pic = Picture()
      pic.folder_id = fid
      pic.path = fp
      pic.filename = os.path.basename(fp)
      pic.width = img.width
      pic.height = img.height
      pic.hwr = 100 * img.height // img.width
      pic.size = os.path.getsize(fp)
      pic.feature_pkl = Feature.featurize(img).to_bytes()
      save(pic)
    return wrapper

  for fn in os.listdir(dp):
    fp = os.path.join(dp, fn)
    if os.path.isfile(fp):
      if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
        continue
      with db_lock:
        if db.query(Picture).filter_by(path=fp).count() == 0:
          scheduler.add_task(tag, _task(fid, fp))
    else:  # isdir(fp)
      if recursive:
        time.sleep(0)   # yield
        index(scheduler, tag, fid, fp, True)

def find_similar_groups(scheduler, tag, pics, sr_mat,
                        sim_thresh, fast_search=True, hwr_tolerance=HWR_TOLERANCE, round_mask=False):
  _N = len(pics)
  us = UnionSet(_N)
  locks = [threading.RLock() for _ in range(_N)]    # the fucking locks
  fv_es = [pic.feature.featvec_edge for pic in pics]
  fv_gs = [pic.feature.featvec_grey for pic in pics] if not fast_search else None

  def _task(idx, idy):
    _d4 = round_mask and 1 or 0
    with locks[idx], locks[idy]:
      if us.in_same_set(idx, idy): return   # FIXME: maybe risk in cocurrency
      if fast_search:
        with db_lock: hwrdiff = abs(pics[idx].hwr - pics[idy].hwr)
        if hwrdiff > hwr_tolerance: return

      sr = sr_mat[idx, idy, 0, _d4]
      if not sr:
        xfv, yfv = fv_es[idx], fv_es[idy]
        if round_mask: xfv, yfv = xfv.round_mask(), yfv.round_mask()
        sr_mat[idx, idy, 0, round_mask] = sr = xfv.similarity_by_avghash(yfv)
      if sr >= sim_thresh: us.union(idx, idy)
      elif not fast_search:
        sr = sr_mat[idx, idy, 1, _d4]
        if not sr:
          xfv, yfv = fv_gs[idx], fv_gs[idy]
          if round_mask: xfv, yfv = xfv.round_mask(), yfv.round_mask()
          sr_mat[idx, idy, 1, round_mask] = sr = xfv.similarity_by_avghash(yfv)
        if sr >= sim_thresh: us.union(idx, idy)
        else:
          sr = sr_mat[idx, idy, 2, _d4]
          if not sr:
            sr_mat[idx, idy, 2, _d4] = sr = xfv.similarity_by_absdiff(yfv)
          if sr >= sim_thresh: us.union(idx, idy)

  task_args = [ ]
  for idx in range(_N - 1):
    for idy in range(idx, _N):
      task_args.append((idx, idy))
  random.shuffle(task_args)
  for args in task_args:
    scheduler.add_task(tag, _task, args=args)

  _cnt = len(task_args)
  if not fast_search: _cnt *= 2
  while scheduler.update_progress(tag, _cnt):
    time.sleep(SLEEP_INTERVAL)
  return [{pics[i] for i in s} for s in us.get_sets(2)]

def filter_pictures(scheduler, tag, pics,
                    hue=None, sim_thresh=0.0, hw_ratio=None, hwr_tolerance=HWR_TOLERANCE):
  ret = [ ]

  def _task(ret, pic):
    if hw_ratio and (pic.hwr < hw_ratio - hwr_tolerance
                     or pic.hwr > hw_ratio + hwr_tolerance):
      return
    if hue and sim_thresh and pic.feature.principle_hues.compability(hue) < sim_thresh:
      return
    ret.append(pic)

  for pic in pics:
    scheduler.add_task(tag, _task, args=(ret, pic))

  _cnt = len(pics)
  while scheduler.update_progress(tag, _cnt):
    time.sleep(SLEEP_INTERVAL)
  return ret

# app.py
def require_folder_selected(func):
  def wrapper(app, *args, **kwargs):
    id = app.ls.curselection()
    if not id: return

    fld = app.folders.get(app.ls.get(id))

    func(app, fld, *args, **kwargs)
  return wrapper

def require_folder_loaded(func):
  def wrapper(app, *args, **kwargs):
    id = app.ls.curselection()
    if not id: return

    fld_name = app.ls.get(id)
    fld = app.folders.get(fld_name)
    pics = app.albums.get(fld_name)
    if pics is None:
      with db_lock:
        pics = db.query(Picture).filter_by(folder=fld).all()
      for pic in pics:
        pkl('load', pic)
        app.pictures[pic.id] = pic
      app.albums[fld.id] = pics

    func(app, fld, *args, **kwargs)
  return wrapper

def require_picture_selected(func):
  def wrapper(app, *args, **kwargs):
    items = app.tv.selection()
    if not items: return

    vals = app.tv.item(items[0], 'values')
    id = vals[0]        # ignore (fn, hw, sz)
    if not id: return   # ignore group title line 'None'
    pic = app.pictures.get(int(id))

    func(app, pic, *args, **kwargs)
  return wrapper

class App:

  def __init__(self):
    logging.debug('[%s] initializing' % self.__class__.__name__)

    self.scheduler = Scheduler(int(os.cpu_count() * WORKER_FACTOR))   # deal with async tasks
    self.cache = LRUCache(CACHE_SIZE)   # for picture/featvec previews, etc...

    self.folders = { }    # { str(fld.name): Folder }
    self.albums = { }     # { int(fld.id): [Picture] }
    self.pictures = { }   # { int(pic.id): Picture }, flattened pool of albums
    self.tv_info = { }    # { int(pic.id): (id, fn, hw, sz) }, info colums in self.tv
    self.tv_view = { }    # { str(fld.id): (str(what)#['listfolder', 'simgrp', 'filter'], Object(data)) }, current tv view of folders
    self.comp_info = { }  # { str(_tag): str(info) }, compare info of selected pictures or hue

    self.setup_gui()
    self.scheduler.set_stat_widgets(self.var_stat_msg, self.pb)
    self.scheduler.add_task('workspace_setup', self.workspace_, args=('setup',))

    logging.debug('[%s] ready' % self.__class__.__name__)
    try:
      tk.mainloop()
    except KeyboardInterrupt:
      pass
    self.workspace_('save')

    self.cache.destroy()
    self.scheduler.stop()
    logging.debug('[%s] exited' % self.__class__.__name__)

  def setup_gui(self):
    # root window
    wnd = tk.Tk()
    wnd.title("Dedup - picture deduplicator (Ver %s)" % __version__)
    (wndw, wndh), scrw, scrh = WINDOW_SIZE, wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    wnd.geometry('%dx%d+%d+%d' % (wndw, wndh, (scrw - wndw) // 2, (scrh - wndh) // 4))
    wnd.resizable(False, False)
    self.wnd = wnd

    # main menu bar
    menu = tk.Menu(wnd, tearoff=False)
    wnd.config(menu=menu)
    self.menu = menu
    if True:
      sm = tk.Menu(menu, tearoff=False)
      sm.add_command(label="Save workspace", command=lambda: self.scheduler.add_task('workspace_save', self.workspace_, args=('save',)))
      sm.add_separator()
      sm.add_command(label="Exit", command=wnd.quit)
      menu.add_cascade(label="File", menu=sm)

      sm = tk.Menu(menu, tearoff=False)
      sm.add_command(label="Open in Explorer", command=lambda: self.fld_open_('explorer'))
      sm.add_command(label="Open in Cmd", command=lambda: self.fld_open_('cmd'))
      sm.add_command(label="Update index", command=self.fld_reindex)
      sm.add_separator()
      sm.add_command(label="Add..", command=self.fld_add)
      sm.add_command(label="Rename", command=self.fld_rename)
      sm.add_command(label="Remove", command=self.fld_remove)
      self.menu_folder = sm
      menu.add_cascade(label="Folder", menu=sm)

      sm = tk.Menu(menu, tearoff=False)
      sm.add_command(label="Preview at left side", command=lambda: self.pic_preview_('L'))
      sm.add_command(label="Preview at right side", command=lambda: self.pic_preview_('R'))
      var = tk.BooleanVar(wnd, PREVIEW_SPLIT)
      self.var_preview_split = var
      sm.add_checkbutton(label='Split preview', variable=var)
      sm.add_separator()
      sm.add_command(label="View image", command=lambda: self.pic_open_('viewer'))
      sm.add_command(label="Locate in Explorer", command=lambda: self.pic_open_('explorer'))
      sm.add_command(label="Open Cmd here", command=lambda: self.pic_open_('cmd'))
      sm.add_command(label="Copy path", command=lambda: self.pic_copy_('path'))
      sm.add_command(label="Copy filename", command=lambda: self.pic_copy_('filename'))
      sm.add_separator()
      sm.add_command(label="View principle hues", command=lambda: self.pic_feature_('phs'))
      sm.add_command(label="View feature vector", command=lambda: self.pic_feature_('fv'))
      sm.add_separator()
      sm.add_command(label="Delete from list", command=lambda: self.pic_delete_('item'))
      sm.add_command(label="Delete file", command=lambda: self.pic_delete_('file'))
      self.menu_picture = sm
      menu.add_cascade(label="Picture", menu=sm)

      sm = tk.Menu(menu, tearoff=False)
      sm.add_command(label="Clear cache", command=lambda: self.scheduler.add_task('cache_clear', self.cache_clear, unique=True))
      sm.add_command(label="Clear all cache", command=lambda: self.scheduler.add_task('cache_clear', self.cache_clear, args=(True,), unique=True))
      sm.add_separator()
      sm.add_command(label="Sanitize database", command=lambda: self.scheduler.add_task('db_sanitize', self.db_sanitize, unique=True))
      sm.add_command(label="Reset database", command=lambda: self.scheduler.add_task('db_reset', self.db_reset, unique=True))
      menu.add_cascade(label="Data", menu=sm)

      menu.add_command(label="Help", command=lambda: tkmsg.showinfo("Help", "I'm sorry but, this fucking world has no help :<"))

    # top: main panel
    frm11 = ttk.Frame(wnd)
    frm11.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, expand=tk.YES)
    if True:
      # left
      frm21 = ttk.Frame(frm11)
      frm21.pack(side=tk.LEFT, fill=tk.BOTH)
      if True:
        # top
        frm32 = ttk.LabelFrame(frm21, text="Settings")
        frm32.pack(side=tk.TOP, fill=tk.X)
        if True:
          # left: labels
          frm41 = ttk.Frame(frm32)
          frm41.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
          if True:
            lb = ttk.Label(frm41, text="sim_thresh")
            lb.pack()

            lb = ttk.Label(frm41, text="fast_search")
            lb.pack()

            lb = ttk.Label(frm41, text="round_mask")
            lb.pack()

            lb = ttk.Label(frm41, text="hw_ratio")
            lb.pack()

            lb = ttk.Label(frm41, text="hwr_tolerance")
            lb.pack()

            lb = ttk.Label(frm41, text="hue")
            lb.pack()

          # right: input widgets
          frm42 = ttk.Frame(frm32)
          frm42.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.YES)
          if True:
            var = tk.DoubleVar(wnd, SIMILAR_THRESHOLD)
            self.var_sim_thresh = var
            sb = tk.Spinbox(frm42, from_=0.5, to=1.0, increment=0.01,
                            justify=tk.CENTER, textvariable=var)
            sb.pack()

            var = tk.BooleanVar(wnd, FAST_SEARCH)
            self.var_fast_search = var
            cb = ttk.Checkbutton(frm42, variable=var)
            cb.pack()

            var = tk.BooleanVar(wnd, ROUND_MASK)
            self.var_round_mask = var
            cb = ttk.Checkbutton(frm42, variable=var)
            cb.pack()

            var = tk.StringVar(wnd, "")
            self.var_hw_ratio = var
            cb = ttk.Combobox(frm42, values=([""] + [i.name for i in HWRatio]),
                              justify=tk.CENTER, textvariable=var)
            cb.pack()

            var = tk.IntVar(wnd, HWR_TOLERANCE)
            self.var_hwr_tolerance = var
            sb = tk.Spinbox(frm42, from_=10, to=25, increment=1,
                            justify=tk.CENTER, textvariable=var)
            sb.pack()

            self.var_hue_rgb = None   # de facto, this is NOT of type tk.Variable, but 3-tuple (R, G, B)
            var = tk.StringVar(wnd, "not selected")
            self.var_hue_hexstr = var
            lb = ttk.Label(frm42, textvariable=var, anchor=tk.CENTER)
            lb.bind('<Button-1>', lambda evt: self._ctl_lb_hue_('choose'))
            lb.bind('<Button-3>', lambda evt: self._ctl_lb_hue_('clear'))
            lb.pack(fill=tk.X, expand=tk.YES)
            self._lb_hue_bg = lb['background']  # save default bg color
            self.lb_hue = lb

        # mid
        frm32 = ttk.LabelFrame(frm21, text="Operation")
        frm32.pack(fill=tk.X)
        if True:
          bt = ttk.Button(frm32, text="List", command=self.op_listfolder)
          bt.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

          bt = ttk.Button(frm32, text="SimGrp", command=self.op_simgrp)
          bt.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

          bt = ttk.Button(frm32, text="Filter", command=self.op_filter)
          bt.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

        # bottom
        frm33 = ttk.LabelFrame(frm21, text="Folders")
        frm33.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
        if True:
          ls = tk.Listbox(frm33, selectmode=tk.SINGLE)
          ls.bind('<Return>', lambda evt: self._ctl_tv_listup())
          ls.bind('<Button-1>', lambda evt: self._ctl_tv_listup())
          ls.bind('<Button-3>', lambda evt: self.menu_folder.post(evt.x_root, evt.y_root))
          ls.pack(fill=tk.BOTH, expand=tk.YES)
          self.ls = ls

      # right
      frm22 = ttk.Frame(frm11)
      frm22.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
      if True:
        # top
        frm33 = ttk.LabelFrame(frm22, text="Pictures")
        frm33.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        if True:
          sb = ttk.Scrollbar(frm33)
          sb.pack(side=tk.RIGHT, fill=tk.Y)

          cols = {
            'id': ("Id", 0, tk.W),
            'fn': ("Filename", 200, tk.W),
            'hw': ("Resolution", 75, tk.CENTER),
            'sz': ("Size", 60, tk.CENTER),
          }
          tv = ttk.Treeview(frm33, show=['headings'],
                            columns=list(cols.keys()), displaycolumns=list(cols.keys())[1:],
                            selectmode=tk.BROWSE, yscrollcommand=sb.set)
          for k, v in cols.items():
            tv.column(k, width=v[1], anchor=v[2])
            tv.heading(k, text=v[0])
          tv.bind('<Double-Button-1>', lambda evt: self.pic_preview_())
          tv.bind('<Return>', lambda evt: self.pic_preview_())
          tv.bind('<Button-3>', lambda evt: self.menu_picture.post(evt.x_root, evt.y_root))
          tv.bind('<Delete>', lambda evt: self.pic_delete_('item'))
          tv.bind('<Shift-Delete>', lambda evt: self.pic_delete_('file'))
          tv.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
          self.tv = tv

        # bottom: picture view
        frm34 = ttk.LabelFrame(frm22, text='Previewer', height=PREVIEW_HW)
        frm34.bind('<Double-Button-1>', lambda evt: self._ctl_pv_('LR', None))
        self.pv = frm34
        if True:
          pv = tk.Label(frm34)
          pv.image, pv.pic = None, None   # object hook against gc
          pv.bind('<Double-Button-1>', lambda evt: self._ctl_pv_('L', None))
          pv.pack(side=tk.LEFT, anchor=tk.CENTER, fill=tk.BOTH, expand=tk.YES)
          self.pvL = pv

          pv = tk.Label(frm34)
          pv.image, pv.pic = None, None   # object hook against gc
          pv.bind('<Double-Button-1>', lambda evt: self._ctl_pv_('R', None))
          pv.pack(side=tk.RIGHT, anchor=tk.CENTER, fill=tk.BOTH, expand=tk.YES)
          self.pvR = pv

          self.pv_show = lambda: self.pv.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
          self.pvL_show = lambda: self.pvL.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
          self.pvR_show = lambda: self.pvR.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.YES)

    # bottom: status bar
    frm12 = ttk.Frame(wnd)
    frm12.pack(side=tk.BOTTOM, anchor=tk.S, fill=tk.X)
    if True:
      var = tk.StringVar(wnd, "Init.")
      self.var_stat_msg = var
      lb = ttk.Label(frm12, textvariable=self.var_stat_msg)
      lb.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

      var = tk.StringVar(wnd, "")
      self.var_info_msg = var
      lb = ttk.Label(frm12, textvariable=self.var_info_msg)
      lb.pack(side=tk.LEFT, anchor=tk.E)

      pb = ttk.Progressbar(frm12)
      pb.pack(side=tk.RIGHT)
      pb.bind('<Button-1>', lambda evt: tkmsg.showinfo("Task scheduler statistics...", self.scheduler.report()))
      self.pb = pb

  def workspace_(self, what):
    if what == 'setup':
      for fld in db.query(Folder).filter_by(deleted=False).all():
        self._rt_indexing(fld)
        self.folders[fld.name] = fld
        self.ls.insert(tk.END, fld.name)

    elif what == 'save':
      for fld in self.folders.values():
        if fld.sr_matrix.modified:
          pkl('save', fld)

  @require_folder_loaded
  def _ctl_tv_listup(self, fld):
    [self.tv.delete(item) for item in self.tv.get_children()]

    with db_lock: fid = fld.id
    what, data = None, None
    kv = self.tv_view.get(fid)
    if kv: what, data = kv

    if data is None:
      self.op_listfolder()
      return

    type = None
    if what in ['listfolder', 'filter']:
      type = 'list'
    elif what == 'simgrp':
      type = 'group'

    _cnt = 0
    if len(data) == 0:
      vals = ('', "[ No appropriate picture found :( ]",)
      self.tv.insert('', 0, values=vals)
    else:
      def _list_group(items, parent):
        for i, pic in enumerate(items):
          id = pic.id
          vals = self.tv_info.get(id)
          if not vals:
            fn = pic.filename
            hw = "%d x %d" % (pic.width, pic.height)
            sz = not (pic.size >> 20) \
                and ("%d KB" % (pic.size >> 10)) \
                or ("%.2f MB" % ((pic.size >> 10) / 1024))
            vals = (id, fn, hw, sz)
            self.tv_info[id] = vals
          self.tv.insert(parent, i, values=vals)

      if type == 'list':
        pics = sorted(data, reverse=True)
        _list_group(pics, '')
        _cnt = len(pics)
      elif type == 'group':
        for i, sim_grp in enumerate(data):
          vals = ('', "[Group-%d]" % (i + 1),)
          grp = self.tv.insert('', i, values=vals, open=True)
          sim_grp = sorted(sim_grp, reverse=True)
          _list_group(sim_grp, grp)
          _cnt += len(sim_grp)

    self.var_info_msg.set('%d pictures listed' % _cnt)

  def _ctl_lb_hue_(self, what='choose'):
    if what == 'choose':
      rgb, hex = tk.colorchooser.askcolor()
      if not rgb or not hex: return

      self.var_hue_rgb = tuple(round(x) for x in rgb)
      self.var_hue_hexstr.set(hex)
      self.lb_hue.config(background=hex, foreground=high_contrast_bw_hexstr(rgb))
    elif what == 'clear':
      self.var_hue_rgb = None
      self.var_hue_hexstr.set("<not selected>")
      self.lb_hue.config(background=self._lb_hue_bg, foreground='#000000')

  def _ctl_pv_(self, which=None, what=None, pic=None):
    if what: self.pv_show()

    if 'R' in which:
      self.pvR.config(image=what)
      self.pvR.image = what
      self.pvR.pic = pic
      if what: self.pvR_show()
      else: self.pvR.pack_forget()
    if 'L' in which:
      self.pvL.config(image=what)
      self.pvL.image = what
      self.pvL.pic = pic
      if what: self.pvL_show()
      else:
        if self.pvR.image:  # could not simply swap because of widgets' side/anchor
          self._ctl_pv_('L', self.pvR.image, self.pvR.pic)
          self._ctl_pv_('R', None)
        else: self.pvL.pack_forget()

    if not self.pvL.image and not self.pvR.image:
      self.pv.pack_forget()

  def _rt_indexing(self, fld):
    with db_lock: fid = fld.id
    _tag = 'op_index-' + str(fld.id)

    def _task():
      with db_lock:
        fld_name = fld.name
        dp = fld.path

      logging.info('[Index] indexing %r (recursive %r)' % (fld_name, RECURSIVE_INDEX))
      index(self.scheduler, ':' + _tag, fid, dp, RECURSIVE_INDEX)

      while self.scheduler.update_progress(_tag):
        time.sleep(SLEEP_INTERVAL)
      pkl('load', fld)
      logging.debug('[Index] indexing %r done' % fld_name)

    self.scheduler.add_task(_tag, _task, unique=True)

  @require_folder_loaded
  def op_listfolder(self, fld):
    with db_lock: fid = fld.id

    data = self.albums.get(fid)
    self.tv_view[fid] = ('listfolder', data)
    self._ctl_tv_listup()

  @require_folder_loaded
  def op_simgrp(self, fld):
    with db_lock: fid = fld.id
    _tag = 'op_simgrp-' + str(fid)

    def _task():
      sim_thresh = self.var_sim_thresh.get()
      fast_search = self.var_fast_search.get()
      hwr_tolerance = self.var_hwr_tolerance.get()
      round_mask = self.var_round_mask.get()

      logging.info('[SimGrp] analyzing %r(%r items) with (sim_thresh %.2f, fast_search %r, round_mask %r)'
                  % (fld.name, len(self.albums.get(fid)), sim_thresh, fast_search, round_mask))
      data = find_similar_groups(self.scheduler, ':' + _tag, self.albums.get(fid), fld.sr_matrix,
                                 sim_thresh=sim_thresh,
                                 fast_search=fast_search, hwr_tolerance=hwr_tolerance,
                                 round_mask=round_mask)

      self.tv_view[fid] = ('simgrp', data)
      self._ctl_tv_listup()

    self.scheduler.add_task(_tag, _task, unique=True)

  @require_folder_loaded
  def op_filter(self, fld):
    with db_lock: fid = fld.id
    _tag = 'op_filter-' + str(fid)

    def _task():
      hue = self.var_hue_rgb
      sim_thresh = self.var_sim_thresh.get()
      hw_ratio = self.var_hw_ratio.get() or None
      if hw_ratio: hw_ratio = HWRatio[hw_ratio].value
      hwr_tolerance = self.var_hwr_tolerance.get()

      logging.info('[Filter] filtering %r(%r items) with (hue %r, sim_thresh %r, hw_ratio %r, hwr_tolerance %r)'
                  % (fld.name, len(self.albums.get(fid)), hue, sim_thresh, hw_ratio, hwr_tolerance))
      data = filter_pictures(self.scheduler, ':' + _tag,
                             self.albums.get(fid), hue=hue, sim_thresh=sim_thresh,
                             hw_ratio=hw_ratio, hwr_tolerance=hwr_tolerance)

      self.tv_view[fid] = ('filter', data)
      self._ctl_tv_listup()

    self.scheduler.add_task(_tag, _task, unique=True)

  @require_folder_selected
  def fld_open_(self, fld, what='explorer'):
    cmd = None
    if what == 'explorer':
      cmd = 'explorer.exe /e,/root,%s' % fld.path
    elif what == 'cmd':
      cmd = 'START /D %s' % fld.path

    if cmd: os.system(cmd)

  @require_folder_selected
  def fld_reindex(self, fld):
    self.scheduler.add_task(None, self._rt_indexing, args=(fld,))

  def fld_add(self):
    dp = tk.filedialog.askdirectory()
    if not dp: return
    dp = dp.replace('/', os.path.sep)

    fld = db.query(Folder).filter_by(path=dp).one_or_none()
    if not fld:
      fld = Folder()
      name, _dir = None, dp
      while not name or name.lower() in FOLDER_NAME_BLACKLIST:
        name, _dir = os.path.basename(_dir), os.path.dirname(_dir)
        if not name: name = '<unknown>'
      fld.name = name
      fld.path = dp
      save(fld)
    elif fld.deleted:
      fld.deleted = False
      save(fld)

    self._rt_indexing(fld)
    self.folders[fld.name] = fld
    self.ls.insert(tk.END, fld.name)

  @require_folder_selected
  def fld_rename(self, fld):
    oldname = fld.name
    newname = tk.simpledialog.askstring('Rename folder..', "New name:")
    if newname: newname = newname.strip(' \t\n\r\0')
    if not newname: return

    fld.name = newname
    save(fld)

    idx = self.ls.curselection()
    self.ls.delete(idx)
    self.ls.insert(idx, newname)
    self.folders[newname] = self.folders.pop(oldname)

  @require_folder_selected
  def fld_remove(self, fld):
    fld.deleted = True
    save(fld)

    self.ls.delete(self.ls.curselection())
    self.folders.pop(fld.name)
    if fld.name in self.pictures:
      self.albums.pop(fld.name)

  @require_picture_selected
  def pic_preview_(self, pic, which=None):
    fp = pic.path
    _key = 'pv-' + str(pic.id)
    img = self.cache.get(_key)
    if not img:
      img = Image.open(fp)
      img = resize_by_hwlimit(img, PREVIEW_HW)
      img = ImageTk.PhotoImage(img)
      self.cache.set(_key, img)

    if not self.var_preview_split.get():
      which = 'L'
      self._ctl_pv_('R', None)  # force clear right side
    elif not which:
      # replace right side has priority
      which = self.pvL.image is None and 'L' or 'R'
    self._ctl_pv_(which, img, pic)

    # show comparing info when both side are normal pictures
    # aka. self._show_comp_info()
    info = ""
    if self.pvL.pic and self.pvR.pic:
      p1, p2 = self.pvL.pic, self.pvR.pic
      _key = '%dx%d' % (min(p1.id, p2.id), max(p1.id, p2.id))
      info = self.comp_info.get(_key)
      if not info:
        _ft1e, _ft2e = p1.feature.featvec_edge, p2.feature.featvec_edge
        _ft1g, _ft2g = p1.feature.featvec_grey, p2.feature.featvec_grey
        info = "[%r, %r] sim_ratio: edge %.2f, grey %.2f/%.2f" % (
          p1.filename[:24],
          p2.filename[:24],
          _ft1e.similarity_by_avghash(_ft2e),
          _ft1g.similarity_by_avghash(_ft2g),
          _ft1g.similarity_by_absdiff(_ft2g),
        )
        self.comp_info[_key] = info
    elif self.var_hue_rgb:
      info = "[%r], hue_comp: %.2f" % (
        pic.filename,
        pic.feature.principle_hues.compability(self.var_hue_rgb),
      )
    self.var_info_msg.set(info)

  @require_picture_selected
  def pic_open_(self, pic, what='viewer'):
    fp = pic.path
    cmd = None

    if what == 'viewer':
      cmd = '"%s"' % fp
    elif what == 'explorer':
      cmd = 'explorer.exe /e,/select,%s' % fp
    elif what == 'cmd':
      cmd = 'START /D %s' % os.path.dirname(fp)

    if cmd: os.system(cmd)

  @require_picture_selected
  def pic_copy_(self, pic, what='path'):
    fp = pic.path
    txt = None

    if what == 'path':       txt = fp
    elif what == 'filename': txt = os.path.basename(fp)

    if txt:
      cpd.OpenClipboard()
      cpd.EmptyClipboard()
      cpd.SetClipboardData(win32con.CF_UNICODETEXT, txt)
      cpd.CloseClipboard()

  @require_picture_selected
  def pic_feature_(self, pic, what='phs'):
    if what == 'phs':
      self.pic_preview_('L')

      _key = 'phs-' + str(pic.id)
      img = self.cache.get(_key)
      if not img:
        img = pic.feature.principle_hues.to_image()
        img = img.resize((PREVIEW_HW, PREVIEW_HW), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.cache.set(_key, img)
      self._ctl_pv_('R', img)

    elif what == 'fv':
      _key = 'fvg-' + str(pic.id)
      img = self.cache.get(_key)
      if not img:
        img = pic.feature.featvec_grey.to_image()
        img = img.resize((PREVIEW_HW >> 1, PREVIEW_HW >> 1), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.cache.set(_key, img)
      self._ctl_pv_('L', img)

      _key = 'fve-' + str(pic.id)
      img = self.cache.get(_key)
      if not img:
        img = pic.feature.featvec_edge.to_image()
        img = img.resize((PREVIEW_HW >> 1, PREVIEW_HW >> 1), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.cache.set(_key, img)
      self._ctl_pv_('R', img)

  @require_picture_selected
  def pic_delete_(self, pic, what='item'):
    fp = pic.path

    if what == 'file':
      if os.path.exists(fp):
        res = shell.SHFileOperation((0, shellcon.FO_DELETE, fp, None,
                                     shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                                     None, None))
        if res[0] == 0: logging.info('[RecycleBin] file %r deleted' % fp)
        self.var_stat_msg.set("Delete %r done." % fp)

    self.tv.delete(self.tv.selection()[0])

  def cache_clear(self, all=True):
    self.tv_view.clear()
    if all:
      self.tv_info.clear()
      self.comp_info.clear()
      self.cache.clear()

  def db_reset(self):
    ok = tk.messagebox.askyesno("Reset db...", "You'll have to rebuild index for all folder! Still sure?")
    if not ok: return

    with db_lock:
      for fld in db.query(Folder).all():
        fld.sr_matrix_pkl = None
        db.add(fld)
      db.commit()

    self.albums.clear()
    self.pictures.clear()
    with db_lock:
      _BATCH_SIZE = 5000
      _ITER = (db.query(Picture).count() + 1) // _BATCH_SIZE
      for i in range(_ITER):
        for pic in db.query(Picture).offset(i * _BATCH_SIZE).limit(_BATCH_SIZE).all():
          pic.feature_pkl = None
          db.add(pic)
        db.commit()

    logging.info('[DB] db_reset done')

    if AUTO_REINDEX:
      for fld in self.folders.values():
        self.fld_reindex(fld)

  def db_sanitize(self):
    logging.info('[DB] db_sanitize started')
    cnt = 0

    # delete ghost folder records
    with db_lock:
      for fld in db.query(Folder).all():
        if not os.path.exists(fld.path):
          cnt += 1
          db.delete(fld)
      db.commit()

    # mark ghost picture as soft deleted
    with db_lock:
      _BATCH_SIZE = 5000
      _ITER = (db.query(Picture).count() + 1) // _BATCH_SIZE
      for i in range(_ITER):
        for pic in db.query(Picture).offset(i * _BATCH_SIZE).limit(_BATCH_SIZE).all():
          if not os.path.exists(pic.path):
            cnt += 1
            pic.deleted = True
        db.commit()

    logging.info('[DB] db_sanitize finished, %d ghost records handled' % cnt)

if __name__ == "__main__":
  App()
