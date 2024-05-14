# Self-Driving Car Model with NVIDIA CNN and Udacity Simulator


## Abstract
This repository contains the implementation of a self-driving car model using NVIDIA's Convolutional Neural Network (CNN) architecture and the Udacity Simulator. The model is trained to autonomously navigate a simulated environment, demonstrating the capabilities of deep learning in the field of autonomous vehicles.

## Introduction
The rapid advancement of deep learning, particularly through the use of convolutional neural networks (CNNs), has dramatically transformed the landscape of autonomous driving technologies. These sophisticated models equip autonomous vehicles with the necessary tools for effectively navigating complex road scenarios and making crucial driving decisions. This project leverages the robust capabilities of CNNs within a virtual environment to simulate human driving behaviors using open-source self-driving car simulators.

Developing an autonomous vehicle involves emulating numerous human driving tasks, including environmental perception, adherence to traffic regulations, and dynamic decision-making. For this project, we utilize CNNs due to their exceptional ability in handling visual data, making them ideal for tasks such as recognizing traffic signals and road signs, as well as understanding complex scenes for navigation. Beyond the realm of machine learning, the development of self-driving cars also demands sophisticated sensors, radars, and hardware components that furnish accurate data inputs to these models.

We have chosen to use an open-source car simulator, focusing solely on the machine learning aspects of autonomous vehicle development, specifically the application of CNNs. These networks are adept at solving critical challenges such as obstacle detection, lane recognition, path planning, and motion control. Safety remains a paramount concern, and previous research has developed CNN models that can effectively interact with erratic drivers, manage unpredictable elements such as pedestrians and unstructured roads, and adhere to complex traffic laws.

The use of simulators in developing autonomous vehicles has been extensively researched. Our choice, the Udacity autonomous car simulator, is well-regarded, although previous studies have highlighted its lack of environmental noise, which could detract from its realism in real-world applications. Despite this, it provides a valuable platform for data acquisition through its training mode, where the vehicle is manually navigated around a track to generate comprehensive datasets, including images from various camera angles and corresponding telemetry data like steering angle, speed, and acceleration.

Leveraging this data, we aim to develop a CNN-based regression model that processes these images to predict accurate steering commands. This approach distinguishes our project from others, including Udacity’s own tutorials, by implementing and evaluating the performance of this CNN model, thus providing an in-depth analysis of its effectiveness in autonomous driving scenarios.



## Analysis

### Data Collection
The data collection process for our self-driving car AI model was meticulously planned to capture a comprehensive range of driving scenarios. We utilized a setup involving three cameras mounted on the vehicle: one front-facing and two side-facing (left and right). This arrangement allowed us to gather diverse visual inputs that are critical for training the model to understand and react to various road conditions and obstacles from multiple perspectives.

#### Challenges in Data Collection
One significant challenge we encountered during data collection was the disproportionate number of samples with a zero steering angle. This issue arises because, during typical driving, the car often moves straight, leading to an overrepresentation of zero steering angle data. Such an imbalance can introduce a bias in the model, where it might overly favor maintaining a straight path regardless of the actual road conditions.

To address this, we implemented a data balancing technique. We identified a horizontal threshold line for zero steering angle data and selectively removed entries above this threshold to balance the dataset. This method ensured that our training data does not disproportionately represent straight-driving scenarios, thereby allowing the model to learn more effectively from turns and curves, which are crucial for real-world driving.

<img width="581" alt="Screenshot 2024-05-14 at 2 59 30 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/83c3ad53-82c4-4f43-b608-afff9ec90b68"><img width="585" alt="Screenshot 2024-05-14 at 3 02 42 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/9aa39061-e7e3-404b-ac55-92ab089bcfa2">


#### Data Splitting Strategy
For the training and validation of our model, the dataset was split into two distinct sets: 80% for training and 20% for validation. It was imperative to split the data in a manner that both subsets reflect a balanced representation of the various steering angles. This balanced distribution is crucial to prevent the model from developing a bias that could skew its performance on either the training or validation set.

Ensuring a consistent distribution of steering angles in both sets helps in evaluating the model’s performance more accurately during the validation phase, providing a reliable measure of how well the model is likely to perform in real-world driving scenarios. This strategy also aids in tuning the model parameters more effectively during the training phase, leading to a robust model capable of generalizing across different driving conditions.

<img width="927" alt="Screenshot 2024-05-14 at 3 04 16 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/6e5a9ddc-7bd7-4865-b6ce-9eccd8f44ab9">

### Applying Some Augmentation Techniques
To enhance the robustness and generalization capability of our self-driving car AI model, we employed a variety of data augmentation techniques. Data augmentation is a critical process in the training of deep learning models, particularly when dealing with visual data, as it helps to artificially expand the size of the training dataset without the need to collect more data. This approach involves applying a series of transformations to the existing data samples to simulate different driving conditions and scenarios. Here are some of the key augmentation techniques we used:

Scaling and Cropping
Scaling and cropping are used to simulate the car being closer or farther from objects in view. This can be particularly useful for teaching the model how different objects might appear at various distances, enhancing its ability to predict appropriate steering angles based on perceived distances. Cropping also helps in focusing the model’s attention on relevant parts of the image, such as the road or obstacles.

<img width="928" alt="Screenshot 2024-05-14 at 3 05 22 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/d2a2fe98-8d2d-4f13-9c62-4173c123745f">


Brightness and Contrast Adjustment
Adjustments to the brightness and contrast of the images simulate different lighting conditions, such as driving at dusk, dawn, or under overcast conditions. This helps ensure that the model remains effective regardless of the time of day or weather conditions, which are common variables that can affect driving visibility and safety.

<img width="906" alt="Screenshot 2024-05-14 at 3 07 36 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/81d67fe8-3063-4424-9511-b14edb07fac8">


Flipping Images Horizontally
This technique effectively doubles the dataset size and is particularly useful for ensuring the model does not develop a bias towards turning more frequently in one direction. By flipping images horizontally, we can simulate the opposite steering direction, thereby teaching the model symmetry in handling left and right turns.

<img width="911" alt="Screenshot 2024-05-14 at 3 08 29 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/80f7592e-d9bc-49da-80a4-ac2136b18105">


Panning
Panning involves horizontally or vertically shifting the image, which is essential for simulating the effect of the vehicle moving within its lane or adjusting to shifts in the road's alignment. This can be particularly useful for teaching the model to handle scenarios where the vehicle needs to adjust its trajectory slightly within the lane to avoid obstacles or to align better with the road. Panning helps increase the variability of input data, ensuring that the model is well-trained to handle minor positional adjustments, a common requirement in real-world driving.

<img width="911" alt="Screenshot 2024-05-14 at 3 09 26 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/309b0322-ef10-497f-90c8-feb05f9b2c7d">


By applying these augmentation techniques, we can artificially create diverse scenarios that help in training a more robust and accurate model. This approach significantly reduces the risk of overfitting, as the model is trained on a broader spectrum of potential real-world conditions, making it more adaptable and reliable when deployed in an actual driving environment.

<img width="1253" alt="Screenshot 2024-05-14 at 3 14 34 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/d6830604-2f01-4417-8458-f9459bdf4d64">


### Data Pre-Processing Techniques
#### Normalization
To ensure uniform lighting and contrast across all images, pixel values were scaled to a range between 0 and 1. This normalization helps reduce variations due to differing light conditions and camera settings.

#### YUV Color Space Conversion
We opted for the YUV color space instead of RGB for processing our images. The YUV format is advantageous because it separates luminance from chrominance, allowing the model to focus more on structural details and less on color variations, which enhances the model's ability to analyze the images.

<img width="1251" alt="Screenshot 2024-05-14 at 3 16 00 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/d4a778d6-9a8f-4b8c-ad81-d19737564ed1">


#### Standardization
To maintain consistency across the dataset, the pixel values were standardized. This process involved adjusting the images so they all have the same mean and standard deviation, facilitating more effective learning by the model as it minimizes scale discrepancies among the input features.

#### Resizing and Smoothing
The images were cropped to exclude unnecessary elements such as the sky and peripheral roadside areas, which do not contribute valuable information for driving decisions. Additionally, a Gaussian blur was applied to smooth out the images, effectively reducing noise and helping to focus the model on important features by softening the details of irrelevant objects.

<img width="1253" alt="Screenshot 2024-05-14 at 3 20 26 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/2057776c-d307-473e-8920-c92d6d6b6767">

## Model

### Model Design

The architecture of our self-driving car model is structured around a deep convolutional neural network (CNN), renowned for its effectiveness in interpreting visual data. The design incorporates several convolutional layers, each layer responsible for detecting and learning a variety of features from the visual inputs. These features range from simple edges and textures in the initial layers to more complex objects and scenarios in deeper layers.

Following the convolutional layers, the model includes multiple fully connected layers. These layers serve to interpret the hierarchical features extracted by the convolutional layers, synthesizing them into a comprehensive understanding that is used to determine the appropriate steering angles. This architecture ensures that each level of abstraction contributes to the final decision-making process in a meaningful way.

### Model Testing
#### Simulation Testing
Our model underwent extensive testing primarily in a simulated environment using the Udacity self-driving car simulator. This approach was selected to ensure comprehensive evaluation under controlled conditions, allowing for the safe testing of the model's response to a variety of driving scenarios. Simulation testing is vital as it allows for risk-free experimentation with extreme cases, such as emergency braking scenarios and complicated dynamic environments, which are challenging to replicate safely in real-world tests.

The model demonstrated excellent performance in these simulations, adapting well to different virtual landscapes and traffic conditions. The rigorous testing in the Udacity simulator included navigating complex urban settings, obeying traffic signals, and reacting to pedestrian crossings, showcasing the model's robust capability to handle diverse driving tasks.

#### Training and Validation
During training, our model achieved impressive results, with both training and validation losses maintained below 0.05 across 10 epochs. This indicates a high level of accuracy in the model's predictions relative to the actual required steering inputs. The low loss values are a strong indicator of the model's ability to generalize from the training data to unseen scenarios in the validation phase, confirming the effectiveness of our data preprocessing and augmentation strategies.

<img width="665" alt="Screenshot 2024-05-14 at 3 26 34 AM" src="https://github.com/AbdelrahmanBadwy/Self_driving_car_model/assets/107257581/bc0fdc7a-5e89-4213-a2e2-e519e40aca54">

In lieu of real-world testing, the extensive simulation-based evaluations provided a solid foundation for assessing the model's practical applicability and safety in real-world conditions, albeit within virtual parameters. The positive results from the Udacity simulator affirm the model's readiness for further developmental stages and potential real-world applications in the future.

## Conclusion
Our project has successfully developed a convolutional neural network (CNN) for autonomous driving, demonstrating significant progress in applying deep learning to complex real-world challenges. Our model, which efficiently processes visual data through advanced architectural design, achieved remarkably low training and validation losses, proving its accuracy and reliability.

Utilizing the Udacity simulator, we rigorously tested the model under various simulated conditions, where it showed excellent adaptability and effectiveness. Although not tested in real-world scenarios, the promising simulation results suggest potential for future on-road applications.

This project enhances our understanding of autonomous driving technologies and sets a solid foundation for further advancements in AI applications within the transportation sector. Overall, we have created a robust, efficient AI-driven system that holds promise for shaping the future of autonomous vehicles.

## Future Work
(Highlight potential enhancements and future directions for the project.)

## License
(Specify the license under which the code and associated resources are distributed.)
## Contributors
Thank you to all the contributors who have helped make Self-Driving Car Model better! ✨
- [Ahmed Khaled](https://github.com/Louda511)
- [Mohamed Magdy](https://github.com/MohamedMagdy097)
- [Abelrahman Ashraf](https://github.com/Abdu117)
- [Rehab Mohamed](https://github.com/rehabmohamed2)
## Acknowledgments
(Give credit to any individuals, organizations, or projects that contributed to the development of the self-driving car model.)

## Contact Information

If you have any questions or feedback, please contact me at [abdoo738@yahoo.com].                          
or my LinkedIn [https://www.linkedin.com/in/abdelrahman-badawy-4517a6231/]

