# GBCNet
This is the official repository for the paper titled "Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning" ([https://arxiv.org/abs/2204.11433](https://arxiv.org/abs/2204.11433)). This paper proposed GBCNet, a specialized CNN model, for classifying gallbladder cancer (GBC) from ultrasound images. GBCNet introduces a novel "multi-scale second-order pooling" block for rich feature encoding from ultrasound images. The paper further proposed a novel visual acuity-based curriculum to train GBCNet. The proposed model beats SOTA deep CNN-based classifiers and human radiologists in classifying GBC from ultrasound images. 

### Installations
We will update the repository soon with the requirements file for the library installations. 

### Model Weights
Download the pre-trained models:
* [GBCNet](https://drive.google.com/file/d/1uBkzlydC9vY8kQsgrlg6983q5WLAK2wS/view)
* [GBCNet with Curriculum](https://drive.google.com/file/d/1s9DMOtgK3TaYBitaYD9IcepmHwxgFZTt/view)

### Dataset
We contributed the first public dataset of 1255 abdominal ultrasound images collected from 218 patients for gallbladder cancer detection. The dataset can be found at: 
[https://gbc-iitd.github.io/data/gbcu](https://gbc-iitd.github.io/data/gbcu)

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
