# pic-dedup
    
    simple picture dedpulicator and by-color filter
    with high accuracy and low speed :lol
    
## Targeted only for Windows system ~

### Quick Start
  - install the requirements.txt
  - run `make dist` to make up [dedup_standalone.py](/dedup_standalone.py), then `make rundist`
    - test demo: run `make rundev`
  - read [settings.py](/dedup/settings.py), for setting references

### Algorithm: compare avghash of featvec
see `Feature` in [imgproc](/dedup/imgproc.py)

### How low speed could it be...
on my laptop (AMD R5 3500U), tested about 1600 pictures:
  - indexing: 243s  (workers = 16, featvec_hw = 16, phs = 4)
  - simgrp: 


by kahsolt
2019/05/20