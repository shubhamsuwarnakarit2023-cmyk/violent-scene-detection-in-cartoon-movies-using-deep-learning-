## Detection of Violent Scenes in Cartoon Movies Using a Deep Learning Approach
 
[![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2024.3480205-blue)](https://doi.org/10.1109/ACCESS.2024.3480205)  


This repository is the ```official open-source``` of [Detection of Violent Scenes in Cartoon Movies
Using a Deep Learning Approach](https://doi.org/10.1109/ACCESS.2024.3480205)
by NOREEN FAYYAZ KHAN, SAREER UL AMIN, ZAHOOR JAN, AND CHANGHUI YAN

## Description
Cartoon movies are a primary source of entertainment for children. However, concerns arise
when these movies inadvertently expose children to violent scenes. This paper addresses the challenge of
detecting subtle instances of violence within cartoon movies. The main difficulty in this task is the sequential
nature of movies, where a sequence of frames must be considered in their given order. Existing methods have
not effectively addressed the issue. In this study, we tackled this challenge by employing a sequential model.
The research comprises three key steps. Initially, a histogram technique was implemented to select key frames
from the video sequences. Subsequently, a Convolutional Neural Network (CNN) was utilized to extract
prominent features from these selected key frames. In the third phase, the acquired features were utilized
to train a sequential model using sequence-based learning. The model was then refined through transfer
learning, using a dataset containing scenes devoid of violence, as well as scenes depicting varying forms
of violence, including bloodshed, fights, gunshots, and explosions. A significant contribution of this study
is the meticulous categorization of violent scenes into four distinct types, allowing for further investigation
into the diverse effects of different violence categories. Furthermore, the study introduces an innovative
approach by integrating a dense layer into the sequential model to enhance final classification. The trained
model’s performance was comprehensively evaluated using metrics such as F1 score, precision, accuracy,
and recall. To validate the effectiveness of the proposed model, it was benchmarked against state-of-theart methods. This study presents an innovative deep-learning methodology for the identification of violent
scenes in cartoon movies. Its potential applications encompass a wide range, including safeguarding children
from inappropriate content.
![Framework](images/fig1.png)

## Prerequisites
- Python 3.9
- keras 2.9.0
- tensorflow-gpu 2.9.1
- CUDA toolkit 11.0
- cuDNN 8.0
- tensorboard 2.9.1
- scikit-learn 1.0.2
- numpy 1.21.6
- opencv 3.4.1
- matplotlib 3.5.3
- seaborn 0.12.2

This method was tested in:
- **GPU**: RTX 3070 with 32 GB memory


## Usage
Clone the repository:
```bash
git clone https://github.com/noreenfayyaz/Detection-of-Violent-Scenes-in-Cartoon-Movies.git
```

## Installation 
To create a conda environment with the required packages, use the following command:
```bash
conda env create -f environment.yml
```

## OurDataset 
 The authors did an extensive search on YouTube to collect a variety of cartoon movies. Our approach to collecting data involved targeted keyword searches and systematic browsing to ensure a comprehensive and diverse dataset. We conducted random browsing to identify a broader range of cartoon content. Specific keywords, such as ‘‘cartoon explosions,’’ ‘‘cartoon fights,’’ ‘‘cartoon gunshots,’’ and ‘‘cartoon bloodshed’’ were used for the cartoon movie search. In addition, this dual approach allowed us to capture a wide variety of violent and non-violent scenes, enhancing the robustness and diversity of our dataset. From these movies, we manually extracted scenes for five distinct categories: Explosion (49 scenes), Bloodshed (46 scenes), Fight (51 scenes), Gunshot (43 scenes), and Normal (53 scenes). Each scene was segmented into shots, which are continuous sequences of frames captured by a single camera from a single view. The duration of these shots ranged from 1 to 5 seconds. The mechanism of frame extraction is illustrated. The number of keyframes extracted from each shot is not fixed and is contingent upon the dissimilarity between the general frames (all frames excluding the reference frame) and the reference frame (typically the first frame of each shot). We set a threshold value for extracting the keyframes: from 1-3 second shots, 15 key frames per shot are extracted, while from 4-5 second shots, 30 key frames per shot are extracted. For validation, the dataset was split into training and testing sets. The training set includes 242 scenes and 6,819 keyframes, while the testing set includes 242 scenes and 2,271 keyframes. This split ensures a balanced representation of each category across both sets.

You can download OurDataset using the download link provided below.

|  File Name |  Download   |   
|:-----------|:-----------|
|OurDataset.zip| [Google Drive](https://drive.google.com/drive/folders/1G4GK1Cw3jmZ5335bO0IJCwf754TD-xjz?usp=sharing)|

```
OurDataset
├── testing
    └── frames
        └── Class 01
            ├── 1.jpg
            ├── ...
            └── n.jpg
        ├── Class 02
        ├── ...
        └── Class 05
└── training
    └── frames
        ├── Class 01
            ├── 1.jpg
            ├── ...
            └── n.jpg
        ├── Class 02
        ├── ...
        └── Class 05
```

![Framework](images/fig2.png)

## Training

To train the model with all default values, use the following script:

```bash
train.ipynb 
```

## Testing

To test and evaluate the model, use the following script:

```bash
testing_model.ipynb 
```
## Qualitative Evaluation
Visualization Results of the proposed method. Overall score of precision, recall and F1-Score for each
experiment.

![Framework](images/fig3.png)

## Citation
If you find this code useful, please cite our paper:
```bash
@ARTICLE{10716639,
  author={Khan, Noreen Fayyaz and Amin, Sareer Ul and Jan, Zahoor and Yan, Changhui},
  journal={IEEE Access}, 
  title={Detection of Violent Scenes in Cartoon Movies Using a Deep Learning Approach}, 
  year={2024},
  volume={12},
  number={},
  pages={154080-154091},
  keywords={Motion pictures;Feature extraction;Visualization;Deep learning;Spatiotemporal phenomena;Explosions;Accuracy;Media;Convolutional neural networks;Computer science;Videos;Classification algorithms;Violence classification;deep learning;animated video classification;sequence learning;shot segmentation},
  doi={10.1109/ACCESS.2024.3480205}}


```

## Acknowledgments
Internal funding has been secured to support this research article, demonstrating the commitment of their institution to advancing knowledge within their academic community. This work used resources of the Center for Computationally Assisted Science and Technology (CCAST) at North Dakota State University, which were made possible in part by NSF MRI Award No. 2019077.

## Contact
```
noreen.f.khan@ndsu.edu
noreen.fayyaz@fu.edu.pk

```
