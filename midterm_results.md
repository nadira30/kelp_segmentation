## Results
#### Method 1: Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
- Upon inspecting the plots of multiple channel separately, as shown in the image below, we can see that the NIR shows the clearest pattern of the kelp canopies. Therefore, we decided to try using the image of this channel to train, validate, and test our CNN model.

![7channel](https://github.com/nadira30/kelp_segmentation/assets/128086407/18671950-b74d-4559-8287-25e96f84c3c4)
|:--:| 
| Image Showing 7 Channels of the satellite TIF image |
  
#### Method 2: Utilizing the Normalized Difference Water Index (NDWI), Normalized Difference Vegetation Index (NDVI), and RGB channels with the U-Net Convolutional Neural Network Architecture.
- The Normalized Difference Water Index (NDWI) is a parameter that may be used to differentiate between different types of vegetation. NDWI = (Near Infrared - Shortwave Infrared)/(Near Infrared + Shortwave Infrared) [6]. Typically, values between -1 and 0 indicate a lack of vegetation or water content, while values greater than 1 indicate the presence of water [6].
- The Normalized Difference Vegetation Index (NDVI) is a parameter that may also be used to differentiate between different types of vegetation. NDVI = (Near Infrared  - Red)/(Near Infrared + Red) [7]. NDVI values typically fall within the range of -1 and +1 with the value increasing in proportion to the increase in vegetation [7]. An NDVI of 0 may indicate a lack of vegetation (e.g. buildings), an NDVI of -1 may indicate a large body of water, and an NDVI of +1 may indicate dense vegetation [7].
- Due to the utility of these parameters in detecting the presence of vegetation, they were used in combination with the RGB channels to train a U-Net model that would be able to return a semantically segmented image with labels corresponding to kelp(1) or no kelp(0).
- Additionally, the digital elevation map values and the cloud mask values were used to filter out irrelevant pixels prior to training. Kelp forests typically extend about 20-30cm above the ocean's surface [8], so pixels with an elevation value that is 30cm's above the ocean's surface were filtered out. Similarly, pixels with the presence of clouds were filtered out as well.

#### Method 3: Using the RGB channel with a modified Resnet Architecture. 
Using the RGB channels as inputs to the convolutional neural network. We can see on the initial RGB image that the kelp is barely noticeable. Therefore we decided to train and test to see how promising is the RGB channels alone to detect kelp. 
![initial image](https://raw.githubusercontent.com/nadira30/kelp_segmentation/nadi_branch/inital.png)
|:--:| 
| Image Showing the 3 RGB channels of the satellite TIF image |

## Results

#### Method1:  Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
- The dataset was split into train-val-test ratio of 70-15-15. The CNN's architecture comprises:
    - Conv2D (32 filters, kernel size (3,3), ReLU activation)
    - MaxPooling2D (2x2)
    - Conv2D (64 filters, kernel size (3,3), ReLU activation)
    - MaxPooling2D (2x2)
    - Conv2D (128 filters, kernel size (3,3), ReLU activation)
    - UpSampling2D (2x2)
    - Conv2D (64 filters, kernel size (3,3), ReLU activation)
    - UpSampling2D (2x2)
    - Conv2D (1 filter, kernel size (3,3), sigmoid activation)
- The visual results and quantitative results are shown in the image below:

![NIR_CNN_results](https://github.com/nadira30/kelp_segmentation/assets/128086407/6ba504be-3bfd-45ac-89be-44b788f1d883)
|:--:| 
| Image Showing results of Method 1: NIR + CNN |


![quant_results_1](https://github.com/nadira30/kelp_segmentation/assets/128086407/c51091cc-a953-4da2-864b-91f04377416f)
|:--:| 
| Image Showing quatitative metric results of Method 1: NIR + CNN |


#### Method 2: Utilizing the Normalized Difference Water Index (NDWI), Normalized Difference Vegetation Index (NDVI), and RGB channels with the U-Net Convolutional Neural Network Architecture.
- Similarly to method 1, the dataset was split into train-val-test ratios of 70-15-15.
- The U-Net CNN architecture is as follows:

|Encoding|Convolution|Decoding|
|-----|-----|-----|
|Conv2D (64 filters, kernel size (3,3))|Conv2D (1600 filters, kernel size (3,3))|Conv2D_transpose (320 filters, kernel size (3,3))|
|Batch Normalization|Batch Normalization|Concatenate|
|ReLU activation|ReLU activation|Conv2D (320 filters, kernel size (3,3))|
|Conv2D (64 filters, kernel size (3,3))| Conv2D (1600 filters, kernel size (3,3))|Batch Normalization|
|Batch Normalization|Batch Normalization|ReLU activation|
|ReLU activation|ReLU activation|Conv2D (320 filters, kernel size (3,3))|
|MaxPooling2D (5x5)||Batch Normalization|
|Conv2D (320 filters, kernel size (3,3))||ReLU activation|
|Batch Normalization||Conv2D_transpose (64 filters, kernel size (3,3))|
|ReLU activation||Concatenate|
|Conv2D (320 filters, kernel size (3,3))||Conv2D (64 filters, kernel size (3,3))|
|Batch Normalization||Batch Normalization|
|ReLU activation||ReLU activation|
|MaxPooling2D (5x5)||Conv2D (64 filters, kernel size (3,3))|
|||Batch Normalization|
|||ReLU activation|
|||Conv2D (2 filters, kernel size (3,3))|


The visual results and quantitative results are shown in the images below:
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/625de30b-2a63-46a3-9cf5-fd90cfff8049)|
|:--:| 
| Image Showing Results of Method 2 |
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/806ef2af-0ebe-4f12-b6c0-ea6855b3c12a)|
|Evaluation of Method 2|

As can be observed by the images and metrics above, method 2 does not perform as well as method 1 and prior implementations [3-6]. This could hint at a need to further tune the parameters used in the model.

#### Method 3: Using the RGB channel with a modified Resnet Architecture. 
- Unlike methods 1 and 2, the dataset was split into train-test rations 80-20.
- The CNN architecture is detailed below.
  *  Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
  * Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
  * Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
  * Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  * Linear(in_features=87616, out_features=128, bias=True)
  * Linear(in_features=128, out_features=128, bias=True)
  * Linear(in_features=128, out_features=245000, bias=True)
  * ReLU()
    
![results_3](https://raw.githubusercontent.com/nadira30/kelp_segmentation/nadi_branch/accuracy.png)
|:--:| 
| Image Showing quatitative metric results of Method 3: CNN |

![quant_results_3](https://raw.githubusercontent.com/nadira30/kelp_segmentation/nadi_branch/Loss%20vs%20Epochs%20lr%3D1e-3.png)
|:--:| 
| Image Showing quatitative metric results of Method 3: CNN |
