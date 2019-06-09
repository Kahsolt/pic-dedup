#!/usr/bin/env python3

import os
import logging
import threading
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base, declared_attr

from . import BASE_PATH, PACKAGE_PATH
from .settings import *

__all__  = ['db', 'db_lock', 'Folder', 'Picture', 'save']

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