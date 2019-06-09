#!/usr/bin/env python3

import os
import re
import datetime
from . import PACKAGE_PATH

def assemble_distro(fout='dedup_standalone.py'):
  _LOCAL_IMPORT_REGEX = re.compile(r'from \..* import .*')
  _FILES = [
    '__init__.py',
    'settings.py',
    'models.py',
    'imgproc.py',
    'utils.py',
    'app.py'
  ]

  lines_import, lines_code = set(), list()
  for fn in _FILES:
    lines_code += ['', '# %s' % fn]
    with open(os.path.join(PACKAGE_PATH, fn)) as fp:
      lastline = ''
      for line in fp.readlines():
        line = line.rstrip(' \n\r\t')
        if not line and not lastline \
            or line.startswith('#!') \
            or line.startswith('__all__') \
            or _LOCAL_IMPORT_REGEX.findall(line):
          continue
        elif line.startswith('import') or line.startswith('from'):
          lines_import.add(line.lstrip(' \t'))
        else:
          lines_code.append(line)
          lastline = line

  with open(fout, 'w+') as fp:
    lines = [
      '#!/usr/bin/env python3',
      '# This file is auto-generated, manual changes should be lost.',
      '# build date: %s.' % datetime.datetime.now(),
      '',
      '__dist__ = "standalone"',  # magic sign for detect_env()
      '',
    ] + sorted(list(lines_import)) + lines_code + [
      '',
      'if __name__ == "__main__":',
      '  App()'
    ]
    for line in lines:
      fp.write(line)
      fp.write('\n')
    fp.flush()

if __name__ == '__main__':
  assemble_distro()