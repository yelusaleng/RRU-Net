# RRU-Net: The Ringed Residual U-Net for Image Splicing Forgery Detection
This repository is for paper ["RRU-Net: The Ringed Residual U-Net for Image Splicing Forgery Detection"](http://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Bi_RRU-Net_The_Ringed_Residual_U-Net_for_Image_Splicing_Forgery_Detection_CVPRW_2019_paper.html)

## Update
upload the pre-trained model.
NOTICING: 
- the uploaded pre-trained model is trained with new datasets since i lost previous pre-trained model.
- the new dataset is produced by my new work, so i can't release it currently.

------

## Requirements
- Python 3.7
- PyTorch 1.0+ 
- CUDA 10.0+

## Details
 - './unet/unet-parts.py': it includes detailed implementations of 'U-Net', 'RU-Net' and 'RRU-Net'
 - 'train.py': you can use it to train your model
 - 'predict.py': you can use it to test

## Citation
Please add following information if you cite the paper in your publication:
```shell
@inproceedings{bi2019rru,
  title={RRU-Net: The Ringed Residual U-Net for Image Splicing Forgery Detection},
  author={Bi, Xiuli and Wei, Yang and Xiao, Bin and Li, Weisheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}
```

Contact yale yalesaleng@gmail.com for any further information.
