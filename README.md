# pic-dedup
    
    simple picture dedpulicator and by-color filter with possibly high accuracy and certainly low speed :lol
    
## Targeted only for Windows system ~

### Screenshots
![Filter by priciple hue](/test/screenshot-0.png)
![Find similar groups](/test/screenshot-1.png)

### Quick Start
  - `pip install pillow numpy sqlalchemy`
  - run normal: `make run` or `python3 dedup_standalone.py`
    - run demo test: `make rundev` or `python3 dedup.py`
  - read [settings.py](/dedup/settings.py) for settings reference

### Algorithm: compare avghash/absdiff of pictures' featvec
see `class Feature` in [imgproc.py](/dedup/imgproc.py)

### Performance: how slow the speed could it be...
on my laptop (AMD R5 3500U, 4c/8p) with 24 workers, testing a folder with 104 pictures (size under 500KB):
  - index: 14s  (featvec_hw = 32, phs = 4)
  - simgrp: 14s (40s without fast_mode, with round_mask)
  - filter: 0.5s


by kahsolt
2019/05/20
