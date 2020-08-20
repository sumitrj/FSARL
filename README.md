# AL & ARL to detect of COVID-19 in Chest Xray images

### Terminology used in the write-up

#### COVID-19 +ve CXR image: 
Chest Xray image of patient of COVID-19
#### COVID-19 -ve CXR image: 
Chest Xray image of suject not suffering from COVID-19, could be normal or be suffering from other diseases that can be detected in Chest Xray images.
#### Normal CXR image: 
Chest Xray image of suject not suffering from any disease that can be detected through Chest Xray image.
#### COVID-19 -ve diseased image:
Chest Xray image of suject not suffering from COVID-19, but suffering from other diseases that can be detected in Chest Xray images.

## Why Active Learning here?

#### a) Class imbalance

Only arouind 400 COVID-19 +ve CXR images and more than 120,000 COVID-19 -ve CXR images. 

#### b) Overfitting

With much lesser images in the postive class, the algorithm may not learn enough features to make accurate classification over unseen data

#### c) Catastrophic forgetting

If transfer learning is used, to learn the features needed to classify from such a diverse variety of features, the model may even perform less efficiently than the base model and result in errors.

## How AL helps in solving these:

### a) Define new framework

Widely accepted solutions for few-shot learning techniques is Transfer Learning.

Pneumonia is found to be very similar to COVID-19 in terms of symptoms, how it affects metabolism, and how it affects the lungs. Since plenty of data is available for this task, we choose the model to classify CXR images into pneumonia and normal as base model. Call it model1.

Split the data of COVID-19 +ve CXR images into different parts in the following percentages:

Total: 100%
Train: 75%
Validate: 25%

#### Set 1:
40% of training data 

#### Set 2:
20% of training data 

#### Set 3:
20% of training data 

#### Set 4:
20% of training data 

Next step is to improvise this model such that it can differentiate between penumonia and other diseases. For this task, we use CXR14 dataset. CXR14 has over 112,000 images including normal images which consist of almost 25-30% of the total dataset. 

We'll split data into 4 sections too. To maintain significance of contributions by different class errors in weighted loss functions, we'll use only 4 times the number of images in the COVID-19 +ve class in the COVID-19 -ve class for each set.

For each of the following steps, train only the decision fumction and the output values and observe difference.

#### Step 1: 
Train over Set 1 of COVID-19 +ve CXR images and randomly select 4 times the number of images in set 1 for COVID-19 -ve class. 

#### Step 2: 

Rank the rest of the images using 4 uncertainty based loss functions. Pick the most uncertain images
and add them to Set 2. Train the model.

Repeat step 2 as step 3 and step 4.

Validate results.

### b) Use Active Reinforcement Learning

In the previous section, we used ranked different data points in decreasing order of uncertainties. Although it may appear that direct ranking of these sample images is a good deterministic solutions, the scale of over hundred thoudand images in pool makes this approach tedious. 

Hence, as a substitute for this ranking algorithm, we need to define an agent which gets rewarded by making the network train over the most uncertain images. The policy that develops overtime must be able to train the network in the best way while being less computationally intensive than the generic sorting algorithm approach which was taken earlier.


## Uncertainty calculation functions:

