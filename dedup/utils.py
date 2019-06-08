#!/usr/bin/env python3

import os
import logging
import numpy as np
import time
import queue
import threading
from collections import defaultdict
from PIL import Image

from .settings import *
from .models import *
from .imgproc import *

__all__ = ['LRUCache', 'Scheduler', 'index', 'find_similar_groups', 'filter_pictures']

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

      # approximately mapping: [0, 100+) => [30, 5]
      next_period = int(next_period ** 2 / -400 + 31)
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
  
  def __init__(self, workers=8):
    self.Q = queue.Queue()
    self.idle = True
    self.evt_stop = threading.Event()
    self.workers = [threading.Thread(
        target=self.__class__.worker, args=(self,),
        name='worker-%d' % (i + 1))
      for i in range(workers)]
    for worker in self.workers: worker.start()

    self.task_starttime = { }                   # { str(tag): int(T) }
    self.task_pending = defaultdict(lambda: 0)  # { str(tag): int(cnt) }
    self.task_finished = defaultdict(lambda: 0) # { str(tag_aggregated): int(cnt) }
    self.bulletin = Scheduler.DummyBulletin()   # cound be reset later, inform use certain progress of tasks
    self.progress_bar = None

    logging.info('[%s] starting with %d workers' % (self.__class__.__name__, workers))

  def set_stat_widgets(self, bulletin, progress_bar):
    self.bulletin = bulletin
    self.progress_bar = progress_bar
  
  def stop(self):
    self.evt_stop.set()
    for worker in self.workers: worker.join()

    logging.debug('[%s] exited' % self.__class__.__name__)

  def add_task(self, tag, task, args=None, unique=False):
    '''
      tag: single-task name, or subtask name (starts with ':')
      task: callable which calls under args
    '''
    if self.idle: self.progress_bar.start()
    
    if not tag: tag = '_' 
    if tag in self.task_pending:
      if unique: return  # ignore on duplicate
    elif tag:
      self.task_starttime[tag] = time.time()
      self.bulletin.set("Task %r started.." % tag)

    self.task_pending[tag] += 1
    self.Q.put((tag, task, args))
  
  def wait_until_done(self, tag, tot_cnt=None):
    time.sleep(SLEEP_INTERVAL)
    rest = self.task_pending.get(tag)
    while rest:
      if tot_cnt:
        self.bulletin.set("Task %s finish %.2f%%." % (tag, 100 * (tot_cnt - rest) / tot_cnt))
      time.sleep(SLEEP_INTERVAL)

  def _task_done(self, tag):
    self.task_pending[tag] -= 1
    if not self.task_pending.get(tag):
      self.task_pending.pop(tag)
      T = time.time() - self.task_starttime.pop(tag)
      self.bulletin.set("Task %s done in %.2fs." % (tag, T))
    
    self.task_finished['-' in tag and tag.split("-")[0] or tag] += 1
    self.Q.task_done()
    
    if not self.Q.unfinished_tasks:
      self.progress_bar.stop()
      self.idle = True
    
  def report(self):
    _nothing = 'Nothing :>'
    pending = '\n'.join([k + ': ' + str(self.task_pending[k]) for k in sorted(self.task_pending)])
    finished = '\n'.join([k + ': ' + str(self.task_finished[k]) for k in sorted(self.task_finished)])
    info = "[Pending]\n" + (pending or _nothing) + "\n\n[Done]\n" + (finished or _nothing)
    return info
     
  @staticmethod
  def worker(scheduler):
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
      if args: task(*args)  # the start * unpacks arguments
      else: task()
      
      scheduler._task_done(tag)

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
  us = UnionSet(len(pics))
  locks = [threading.RLock() for _ in range(len(pics))]    # the fucking locks
  fv_es = [pic.feature.featvec_edge for pic in pics]
  fv_gs = [pic.feature.featvec_grey for pic in pics] if not fast_search else None

  def _task(idx, xfv, idy, yfv):
    with locks[idx], locks[idy]:
      sr = sr_mat[idx, idy, 0]
      if not sr: sr_mat[idx, idy, 0] = sr = xfv.similarity_by_avghash(yfv)
      if sr >= sim_thresh: us.union(idx, idy)
      elif not fast_search:
        sr = sr_mat[idx, idy, 1]
        if not sr: sr_mat[idx, idy, 1] = sr = fv_gs[idx].similarity_by_avghash(fv_gs[idy])
        if sr >= sim_thresh: us.union(idx, idy)
        else:
          sr = sr_mat[idx, idy, 2]
          if not sr: sr_mat[idx, idy, 2] = sr = fv_gs[idx].similarity_by_absdiff(fv_gs[idy])
          if sr >= sim_thresh: us.union(idx, idy)

  for idx, xfv in enumerate(fv_es):
    for idy, yfv in enumerate(fv_es):
      if idx >= idy or us.in_same_set(idx, idy): continue
      if fast_search:
        with db_lock:
          hwrdiff = abs(pics[idx].hwr - pics[idy].hwr)
        if hwrdiff > hwr_tolerance: continue
      if round_mask: xfv, yfv = xfv.round_mask(), yfv.round_mask()
      
      scheduler.add_task(tag, _task, args=(idx, xfv, idy, yfv))
  
  _cnt = len(pics) * (len(pics) - 1) / 2
  if not fast_search: _cnt *= 3
  scheduler.wait_until_done(tag, _cnt)
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

  scheduler.wait_until_done(tag, len(pics))
  return ret
