# Codes-for-Deep-Transfer-Learning-Based-Downlink-Channel-Prediction-for-FDD-Massive-MIMO-Systems
Data-Generation-and-Codes-for-Deep-Transfer-Learning-Based-Downlink-Channel-Prediction-for-FDD-Massive-MIMO-Systems

## Notice
In order to use the DeepMIMO datasets/codes or any (modified) part of them, please cite
1. The corresponding paper: Yang, Y., Gao, F., Zhong, Z., Ai, B., A. Alkhateeb. (2019). [Deep Transfer Learning Based Downlink Channel Prediction for FDD Massive MIMO Systems](https://arxiv.org/abs/1912.12265).
2. The Remcom Wireless InSite website: Remcom, [Wireless insite](https://www.remcom.com/wireless-insite).
<details>
<summary>Unfold to see bibtex codes</summary>
<pre><code>
@article{yang2019deep,
  title={Deep Transfer Learning Based Downlink Channel Prediction for FDD Massive MIMO Systems},
  author={Y. Yang and F. Gao and Z. Zhong and B. Ai  and A. Alkhateeb},
  journal={arXiv preprint arXiv:1912.12265},
  year={2019}
}
@unpublished{timmurphy,
title={Remcom Wireless InSite},
note = {\url{https://www.remcom.com/wireless-insite-em-propagation-software}}
}
</code></pre>
</details>

## Dependencies
This code requires the following: python 3.*, TensorFlow v1.4+


## Usage instructions

They also provide the foundation to reproduce the other results 
### Data Generation
To access the datasets (i.e., samples_target64_1036_2.mat and samples_source64_1552_2.mat), please click [here](https://drive.google.com/drive/folders/17WBUbbqnLbUjTuMuGNgnddwCx5Uw_q0H?usp=sharing).


More questions about the data generation, please contact: yyw18@mails.tsinghua.edu.cn.

## Reproduce the Demo Result

Reproducing The Figure:
Generate a dataset for scenario I1_2p4 using the settings in the table above--number of paths should be 1.
Organize the data into a MATLAB structure named "rawData" with the following fields: channel and userLoc. "channel" is a 3D array with dimensions: # of antennas X # of sub-carriers X # of users while "userLoc" is a 2D array with dimensions: 3 X # of users.
Save the data structure into a .mat file.
In the file main, set the option: options.rawDataFile1 to point to the .mat file.
Run main.m

## Related sources

## Not finished yet, please wait for updates!
