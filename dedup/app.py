#!/usr/bin/env python3

import os
import logging
import win32con
import win32clipboard as cpd
from win32com.shell import shell, shellcon
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.colorchooser
import tkinter.messagebox as tkmsg

from . import __version__
from .settings import *
from .models import *
from .imgproc import *
from .utils import *

__all__ = ['App']

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

    self.folders = { }  # { str(fld.name): Folder }
    self.albums = { }   # { int(fld.id): [Picture] }
    self.pictures = { } # { int(pic.id): Picture }, flattened pool of albums
    self.tv_info = { }  # { int(pic.id): (id, fn, hw, sz) }, info colums in self.tv
    self.tv_view = { }  # { str(fld.id): (str(what)#['listfolder', 'simgrp', 'filter'], Object(data)) }
    self.comp_info = { }# { str(pic1.id + pic2.id): str(info) }, compare info of two pictures
    
    self.setup_gui()
    self.scheduler.set_stat_widgets(self.var_stat_msg, self.pb)
    self.scheduler.add_task('workspace_setup', self.workspace_, args=('setup',))
    logging.debug('[%s] ready' % self.__class__.__name__)
    try:
      tk.mainloop()
    except KeyboardInterrupt:
      pass
    self.scheduler.add_task('workspace_save', self.workspace_, args=('save',))

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
      sm.add_command(label="Update index", command=lambda: self.fld_reindex)
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
      sm.add_command(label="Sanitize database", command=lambda: self.scheduler.add_task('db_sanitize', sanitize_db, unique=True))
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
      for fld in self.folders:
        pkl('save', fld)


  @require_folder_loaded
  def _ctl_tv_listup(self, fld):
    [self.tv.delete(item) for item in self.tv.get_children()]
    
    fid = fld.id
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
      elif type == 'group':
        for i, sim_grp in enumerate(data):
          vals = ('', "[Group-%d]" % (i + 1),)
          grp = self.tv.insert('', i, values=vals, open=True)
          sim_grp = sorted(sim_grp, reverse=True)
          _list_group(sim_grp, grp)

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
    fid = fld.id
    _tag = 'op_index-' + str(fld.id)

    def _task():
      with db_lock:
        fld_name = fld.name
        dp = fld.path
          
      logging.info('[Index] indexing %r (recursive %r)' % (fld_name, RECURSIVE_INDEX))
      index(self.scheduler, ':' + _tag, fid, dp, RECURSIVE_INDEX)

    self.scheduler.add_task(_tag, _task, unique=True)
    self.scheduler.wait_until_done(_tag)
    pkl('load', fld)


  @require_folder_loaded
  def op_listfolder(self, fld):
    fid = fld.id
    _tag = 'op_listfolder-' + str(fid)

    def _task():
      data = self.albums.get(fid)
    
      self.tv_view[fid] = ('listfolder', data)
      self._ctl_tv_listup()
    
    self.scheduler.add_task(_tag, _task, unique=True)
  
  @require_folder_loaded
  def op_simgrp(self, fld):
    fid = fld.id
    _tag = 'op_simgrp-' + str(fid)

    def _task():
      sim_thresh = self.var_sim_thresh.get()
      fast_search = self.var_fast_search.get()
      hwr_tolerance = self.var_hwr_tolerance.get()
      round_mask = self.var_round_mask.get()

      logging.info('[SimGrp] analyzing %r with (sim_thresh %.2f, fast_search %r, round_mask %r)' 
                  % (fld.name, sim_thresh, fast_search, round_mask))
      data = find_similar_groups(self.scheduler, ':' + _tag, self.albums.get(fid), fld.sr_matrix,
                                 sim_thresh=sim_thresh, 
                                 fast_search=fast_search, hwr_tolerance=hwr_tolerance,
                                 round_mask=round_mask)
      pkl('save', fld)
      
      self.tv_view[fid] = ('simgrp', data)
      self._ctl_tv_listup()
    
    self.scheduler.add_task(_tag, _task, unique=True)

  @require_folder_loaded
  def op_filter(self, fld):
    fid = fld.id
    _tag = 'op_filter-' + str(fid)

    def _task():
      hue = self.var_hue_rgb
      sim_thresh = self.var_sim_thresh.get()
      hw_ratio = self.var_hw_ratio.get() or None
      if hw_ratio: hw_ratio = HWRatio[hw_ratio].value
      hwr_tolerance = self.var_hwr_tolerance.get()

      logging.info('[Filter] filtering %r with (hue %r, sim_thresh %r, hw_ratio %r, hwr_tolerance %r)'
                  % (fld.name, hue, sim_thresh, hw_ratio, hwr_tolerance))
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
          p1.filename,
          p2.filename,
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
