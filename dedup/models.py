#!/usr/bin/env python3

import os
import logging
import threading
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base, declared_attr

from . import BASE_PATH, PACKAGE_PATH
from .settings import *

__all__  = ['db', 'db_lock', 'Folder', 'Picture', 'sanitize_db', 'save']

db = None
db_lock = threading.RLock()
Model = declarative_base()

class Folder(Model):

  @declared_attr
  def __tablename__(cls): return cls.__name__

  id = sql.Column(sql.INTEGER, primary_key=True, autoincrement=True)

  path = sql.Column(sql.TEXT, unique=True, comment='absolute path')
  name = sql.Column(sql.TEXT, comment='folder basename or distinguishable human readable name')
  deleted = sql.Column(sql.BOOLEAN, default=False, comment='soft delete mark')

  sr_matrix_pkl = sql.Column(sql.BLOB, comment='pickled sim_ratio matrix')
  sr_matrix = None  # instance of np.ndarray, loaded from sr_matrix_pkl
  
  pictures = relationship('Picture', order_by='Picture.id', back_populates='folder', lazy='dynamic')

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
  hwr = sql.Column(sql.INT, comment='hwr * 100')
  size = sql.Column(sql.INT, comment='file size in bytes')
  deleted = sql.Column(sql.BOOLEAN, default=False, comment='soft delete mark, keeping rank in sr_matrix')
  
  feature_pkl = sql.Column(sql.BLOB, comment='pickled Feature')
  feature = None  # instance of Feature, loaded from feature_pkl

  def __repr__(self):
    return '<%s id=%r folder=%r path=%r width=%r height=%r, size=%r>' % (
        self.__class__.__name__, self.id, self.folder, self.path, self.width, self.height, self.size)

  def __lt__(self, other):
    return self.size < other.size
  
def setup_db(dbname='index.db'):
  global db

  # detect env
  if globals().get('__dist__') == 'standalone':
    _env = 'dist'
  elif os.getenv('DEDUP_ENV') == 'test':
    _env = 'test'
  else:
    _env = 'dev'

  # configure database file
  if _env == 'test':
    dbfile = os.path.join(BASE_PATH, 'test', dbname)
  else:
    if DB_FILE:
      dbfile = DB_FILE
    elif _env == 'dist':
      dbfile = os.path.join(PACKAGE_PATH, dbname)
    else:
      dbfile = os.path.join(BASE_PATH, dbname)
  logging.info('[DB] use %s' % dbfile)

  # orm session
  engine = sql.create_engine('sqlite:///%s?check_same_thread=False' % dbfile)
  engine.execute('PRAGMA journal_mode = PERSIST;').fetchall()
  Model.metadata.create_all(engine)
  session_maker = sessionmaker(bind=engine)
  db = session_maker()

  # test env auto setup hook
  if _env == 'test':
    logging.info('[DB] test env setup & init...')
    dp = os.path.dirname(dbfile)
    if db.query(Folder).filter_by(path=dp).count() == 0:
      fld = Folder()
      fld.name = os.path.basename(dp)
      fld.path = dp
      save(fld)

def sanitize_db():
  logging.info('[DB] sanitize_db started')
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

  logging.info('[DB] sanitize_db finished, %d ghost records handled' % cnt)

def save(model=None):
  with db_lock:
    if model: db.add(model)
    try: db.commit()
    except Exception as e: logging.error(e)

# global initialize
logging.basicConfig(level=LOG_DEBUG and logging.DEBUG or logging.INFO, 
                    format="%(asctime)s - %(levelname)s: %(message)s")
setup_db()