# Convolutional Character Networks

> **Modified version** — this is an adapted copy of the original CharNet repository.  
> See [NOTICE](NOTICE) for a full list of changes from the original source.  
> Original repository: <https://github.com/MalongTech/research-charnet>

This project hosts the testing code for CharNet, described in our paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

   
## Installation

```
pip install torch torchvision
python setup.py build develop
```


## Run
1. **Download pretrained weights.**
   The original CharNet repository hosted the weights on cloudstor, but the
   link has become unreliable. We provide the script below as a convenience,
   but if it fails you can obtain the file from the following GitHub issue
   where a user has mirrored it:
   <https://github.com/HilaManor/Scene-Understanding-Based-on-Text-Extraction/issues/24>
   
   ```bash
   bash download_weights.sh      # saves weights/icdar2015_hourglass88.pth
   ```
   
   Alternatively, download `icdar2015_hourglass88.pth` manually and place it
   in this directory under `weights/` (the config value `WEIGHT` points there
   and paths are resolved automatically by `resolve_charnet_paths`).

2. For ICDAR 2015, run the network on an image directory. Replace
   `images_dir` with the path to your images and `results_dir` with where you
   want JSON outputs saved.

    ```
    python tools/test_net.py configs/icdar2015_hourglass88.yaml <images_dir> <results_dir>
    ```


## Citation

If you find this work useful for your research, please cite as:

    @inproceedings{xing2019charnet,
    title={Convolutional Character Networks},
    author={Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
    }
    
## Contact

For any questions, please feel free to reach: 
```
github@malongtech.com
```


## License

CharNet is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.
