# GBCNet
This is the official repository for the paper titled "Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning" ([https://arxiv.org/abs/2204.11433](https://arxiv.org/abs/2204.11433)). This paper proposed GBCNet, a specialized CNN model, for classifying gallbladder cancer (GBC) from ultrasound images. GBCNet introduces a novel "multi-scale second-order pooling" block for rich feature encoding from ultrasound images. The paper further proposed a novel visual acuity-based curriculum to train GBCNet. The proposed model beats SOTA deep CNN-based classifiers and human radiologists in classifying GBC from ultrasound images. 

### Installations
We will update the repository soon with the requirements file for the library installations. 

### Model Weights
Download the pre-trained models:
* [GBCNet](https://drive.google.com/file/d/1yzimpNKLr5z04CUZA7ls7HkEECZc5JQ7/view)
* [GBCNet with Curriculum](https://drive.google.com/file/d/1d3T-n4LdpeuBgw9CxVxpGf6IAb1iiIAm/view)
* [Initial weights (warmup ckpt, use if you're training)](https://drive.google.com/file/d/1sJ6DflfoMMkM9lewlq5k2fna3SKFHF8r/view)

### Dataset
We contributed the first public dataset of 1255 abdominal ultrasound images collected from 218 patients for gallbladder cancer detection. The dataset can be found at: 
[https://gbc-iitd.github.io/data/gbcu](https://gbc-iitd.github.io/data/gbcu)

### ROI Detection
The FasterRCNN-based ROI detection model code and weight is available in [this link](https://drive.google.com/file/d/1E_LoLKjZ1Co-HrAcPbDasHpDXrJ3Caw2/view). 
The output of this model is already stored in the `roi_pred.json` file in the dataset.

### Citation
```
@inproceedings{basu2022surpassing,
  title={Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning},
  author={Basu, Soumen and Gupta, Mayank and Rana, Pratyaksha and Gupta, Pankaj and Arora, Chetan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20886--20896},
  year={2022}
}
```
### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)  
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial use only. Any commercial use should obtain formal permission.

### Acknowledgement
This code base is built upon [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels), [MPN-COV](https://github.com/jiangtaoxie/MPN-COV), and [GSoP](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks). Thanks to the authors of these papers for making their code available for public usage.  
