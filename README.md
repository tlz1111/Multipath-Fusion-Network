# Multipath-Fusion-Network
Main code of paper "Human Activity Recognition Based on Multipath Fusion in Non-line-of-sight Corner"

Abstract:
Radar-based human activity recognition (HAR) holds significant application value in fields such as medical rehabilitation and security monitoring. However, existing recognition methods primarily address line-of-sight (LOS) and non-line-of-sight through-wall (NLOS-TW) scenarios, neglecting consideration for non-line-of-sight corner (NLOS-C) scenario within urban architecture. In NLOS-C scenario, electromagnetic waves illuminate the target through multiple paths, resulting in considerable variations in range-time map, causing performance degradation or even failure of existing methods. Moreover, multipath propagation enables a single-node radar to function equivalently as a multi-perspective multi-node radar system, thereby providing richer and more comprehensive information for human activity. Therefore, considering the complementary interpretations of multipath on target behavior and the distinctive features observed in NLOS-C range-time map, this paper proposes a HAR method based on multipath fusion to accurately identify human behaviors in NLOS-C scenario. Firstly, considering the broad distribution and large span characteristics of behavior features in the range dimension caused by multipath effect, we design a multipath information fusion module based on dilated convolution to effectively integrate and interact the multipath information. Additionally, to address the diverse feature scales caused by variable widths and blurred boundaries of each path, we incorporate multi-scale unit into the deep feature extraction module to enhance the capability of autonomously adjusting receptive field. Finally, multipath interaction information is fused with depth features for behavior judgment. Experimental results validate the effectiveness of the proposed method in NLOS-C scenario.

# Install and compile the prerequisites
- Python 3.9
- Pytorch 2.2.2
- NVIDIA GPU + CUDA
- Python packages: numpy, torchvision, PIL

# Test
The test data is provided in the file **"test data.zip"**, and the pre-trained model parameters are available in **"model.zip"**. To run the test, simply update the data and model paths in **test.py** accordingly.

