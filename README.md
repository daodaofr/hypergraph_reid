# hypergraph_reid

Implementation of "Learning Multi-Granular Hypergraphs for Video-Based Person Re-Identification"
If you find this help your research, please cite

    @inproceedings{DBLP:conf/cvpr/YanQC0ZT020,
      author    = {Yichao Yan and
                   Jie Qin and
                   Jiaxin Chen and
                   Li Liu and
                   Fan Zhu and
                   Ying Tai and
                   Ling Shao},
      title     = {Learning Multi-Granular Hypergraphs for Video-Based Person Re-Identification},
      booktitle = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                   {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
      pages     = {2896--2905},
      publisher = {{IEEE}},
      year      = {2020}
    }


## Installation
We use python 3.7 and pytorch=0.4

## Data preparation
All experiments are done on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```

### Usage
To train the model, please run
sh run_hypergraphsage_part.sh

### Performance
Normaly the model achieves 85.8%  mAP and 89.5% rank-1 accuracy. According to my training log, the best model achieves 86.2% mAP and 90.0% top-1 accuracy. This may need adjustion in hyperparameters.

### Acknowledgements
Our code is developed based on Video-Person-ReID (https://github.com/jiyanggao/Video-Person-ReID). 
