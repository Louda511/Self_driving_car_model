


## Abstract
This repository contains the implementation of a self-driving car model using NVIDIA's Deep Neural Network (DNN) architecture and the Udacity Simulator. The model is trained to autonomously navigate a simulated environment, demonstrating the capabilities of deep learning in the field of autonomous vehicles.

## Introduction
Self-driving cars represent a significant advancement in the field of artificial intelligence and robotics. By leveraging deep learning techniques, we can train models to perceive and interpret their surroundings, make decisions, and navigate safely through various environments. This project aims to showcase one such implementation using NVIDIA's DNN architecture in conjunction with the Udacity Simulator.

## Methods

### Data Sourcing
The training data was sourced from Udacity's autonomous car simulator, which provides two tracks with varying complexities. While the simulator lacks features like traffic signs and pedestrians, it offers scenarios for turning, speeding, and slowing down.

### Data Augmentation
#### Balancing the Number of Turns
Due to the absence of right turns in one of the simulator's tracks, we augmented the data by flipping a random set of images and negating the steering angle to account for the lack of balance.

```python
# Function to balance right and left turns
def balance(image, angle):
    image = cv2.flip(image, 1)
    angle = -angle
    return image, angle
```

Additionally, we balanced the distribution of steering angles by deleting a randomized set of data corresponding to driving straight. This ensured a more uniform distribution of data.

#### Noise
To enhance model robustness, we added noise through randomized rotations, shifts, and blurs. Changes in brightness were also applied to simulate varying lighting conditions, aiding the model's adaptability.

### Data Pre-Processing
#### Normalization
Pixel values were normalized to the range [0, 1] to mitigate variations in lighting, contrast, and color.

#### YUV Color Space
YUV color space was chosen over RGB to separate color and brightness information, facilitating more efficient image analysis.

#### Standardization
Pixel values were standardized to have a consistent mean and standard deviation, enhancing the comparability and processability of images.

#### Resize and Gaussian Blur
Images were cropped to remove irrelevant features such as the sky and road verge. Gaussian blur was applied to reduce noise and smooth out details.

## Discussion - Model
(Add your discussion here based on your model's performance, challenges faced, and potential improvements.)

## Usage
(Provide instructions for setting up the environment, installing dependencies, and running the model in the Udacity Simulator.)

## Future Work
(Highlight potential enhancements and future directions for the project.)

## License
(Specify the license under which the code and associated resources are distributed.)
## Contributors
Thank you to all the contributors who have helped make Self-Driving Car Model better! ✨
- [Ahmed Khaled](https://github.com/Louda511)
- [Mohamed Magdy](https://github.com/MohamedMagdy097)
- [Abelrahman Ashraf](https://github.com/Abdu117)
- [Rehab Mohamed]()
## Acknowledgments
(Give credit to any individuals, organizations, or projects that contributed to the development of the self-driving car model.)

## Contact Information

If you have any questions or feedback, please contact me at [abdoo738@yahoo.com].                          
or my LinkedIn [https://www.linkedin.com/in/abdelrahman-badawy-4517a6231/]

