# Kelp Segmentation: Mapping and Monitoring of Kelp Forests
Computer Vision Class project: https://www.drivendata.org/competitions/255/kelp-forest-segmentation/data/

### Introduction/Problem Definition:

A kelp forest is an ocean community made up of dense groupings of kelps. These forests typically serve as a source of food, shelter, and protection to a wide variety of marine life [1]. Moreover, the significance of kelp forests extends beyond their role as a biodiverse hotspot. They play a pivotal role in regulating marine environments and supporting human well-being. Kelps are renowned for their capacity to sequester carbon dioxide through photosynthesis, contributing significantly to carbon capture from human's daily life and carbon storage in the oceans [2]. Additionally, these underwater havens act as natural buffers against coastal erosion, mitigating the impacts of waves and currents on shorelines. Furthermore, kelp forests hold economic value, serving as crucial fishing grounds and tourist attractions in many coastal regions.

However, these forests face threats from multitude of factors such as climate change, acidification, overfishing, and unsustainable harvesting practices [1]. 

To address these pressing conservation concerns and safeguard the future of kelp forests, innovative approaches are needed. One promising strategy involves harnessing the power of technology to monitor and protect these vital marine habitats. By leveraging advances in remote sensing, computer vision, and machine learning, we propose the development of a comprehensive model capable of monitoring kelp forests using coastal satellite imagery. Such a model would enable tracking of changes in kelp abundance, empowering conservation efforts and informing sustainable management practices.

| ![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/965b146e-3d3b-4394-955c-e50391a7ab1c) | 
|:--:| 
| *Image Showing Sardines Seeking Food and Shelter in a Kelp Forest [2]*|

![kelpswir_oli_2019](https://github.com/nadira30/kelp_segmentation/assets/128086407/7d7aa4ab-a17f-45c3-815f-8f9814d8d7c5)
|:--:| 
| *Image Showing the change in Kelp forest abundance from 2008 to 2019* |

### Related Works: 

#### [3] Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas):
- The paper evaluates two methods to automate kelp detection from satellite imagery: 1) crowdsourced classifications from the Floating Forests citizen science project (FF8), and 2) an automated spectral approach using a decision tree combined with multiple endmember spectral mixture analysis (DTM).
- Both methods were applied to classify kelp from Landsat 5, 7, 8 imagery covering the Falkland Islands from 1985-2021.
- DTM showed better performance than FF8 when validated against expert manual classifications of 8 Landsat scenes covering over 2,700 km of coastline.
- Multiple Endmember Spectral Mixture Analysis (MESMA) is a spectral unmixing algorithm that estimates the fractional contributions of pure spectral endmembers (e.g. water, kelp, glint) to each image pixel spectrum based on a linear mixing model. It allows estimating partial kelp coverage within pixels.
- Decision Tree Classification is used to first identify potential candidate kelp-containing pixels before applying MESMA. The decision tree uses spectral rules to separate kelp from non-kelp pixels. Then MESMA spectral unmixing is utilized to estimate fractional kelp coverage within those candidate pixels
  
![results_remote_sensing](https://github.com/nadira30/kelp_segmentation/assets/128086407/bd4350ca-dd28-45f5-9a77-abd99fae646c)
|:--:| 
| Image Showing the results of automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas) |

#### [4] Mapping bull kelp canopy in northern California using Landsat to enable long-term monitoring
- This paper is focused on the mapping and monitoring of kelp, specifically bull kelp, though the use of Landsat satellite images.
- Similar to our previous reference, MESMA was used to predict the presence of bull kelp biomass.
- This paper discusses the influence of water endmember quantity and the use of giant kelp versus bull kelp endmembers on the accuracy of MESMA; tidal influence on bull kelp canopy areas; and performs a comparison of the use of Landsat images versus other surveys. The researchers were able to conclude that:
  - The number of water endmembers does not significantly influence MESMA performance.
  - Bull kelp specific endmembers outperform repurposed giant kelp endmembers.
  - Tidal phases do not significantly influence the detection of bull kelp canopy areas.
  - Compared to aerial imagery and private satellite images that have been employed by other researchers, Landsat images allow for long-term montioring since they are updated by the U.S. Geological Survey (USGS) every 16 days.

#### [5] Automatic Hierarchical Classification of Kelps Using Deep Residual Features

This paper presents a binary classification method that classifies kelps in images collected by autonomous underwater vehicles. 
The paper shows that kelp classification using classification by deep residual features DRF outperforms CNN and features extracted from pre-trained CNN such as ImageNets. The performance was demonstrated using ground truth data provided by marine experts and showed a high correlation with previously conducted manual surveys. The metrics evaluated were: F1 score, accuracy, precision, and recall.  
A binary classifier is trained for every node in the hierarchical tree of the given problem and deep residual networks (ResNets) are extracted to improve time efficiency and the automation process of detection of kelp marine species.

Furthermore, color channel stretch was used on images to reduce the effect of the underwater color distortion phenomenon. For feature extraction, a pre-trained Resnet 50 was used, and the proposed method was implemented using MatConvNet and the SVM classifier.
Despite DRF allowing the comparison of kelp coverage in different sites, the proposed method had the drawback of an over-prediction of kelp at high percentage cover and under-prediction at low coverage, even though the prediction was negligible in some sites. 

### Methods/Approach: 

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

### Experiments / Results: 

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

  
### What’s next: 

|Task | Date|
|-----|-----|
|Select the most promising CNN architecture, fine-tune| April 06th|
|Use different combinations and variations of inputs before training and testing| April 10th|
|Determine most applicable metrics| April 12th|
|wrap up project and report| April 16th|


### Team member contributions: 
Nadira:
- Formatting dataset 
- Develop a custom CNN architecture
- Training and testing method 3

Tofunmi:
- Formatting Dataset 
- Develop a UNET CNN architecture
- Training and testing method 2

Thanapol Tantagunninat:
- Formatting and splitting the common training/testing dataset
- Develop the CNN architecture
- Training and testing the method 1


### References
[1] Kelp Forest. Kelp Forest | NOAA Office of National Marine Sanctuaries. (n.d.). https://sanctuaries.noaa.gov/visit/ecosystems/kelpdesc.html 

[2] Browning, J., & Lyons, G. (2020, May 27). 5 reasons to protect kelp, the West Coast’s powerhouse Marine Algae. The Pew Charitable Trusts. https://www.pewtrusts.org/en/research-and-analysis/articles/2020/05/27/5-reasons-to-protect-kelp-the-west-coasts-powerhouse-marine-algae#:~:text=3.-,Protect%20the%20shoreline,filter%20pollutants%20from%20the%20water. 

[3] Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas)
Houskeeper HF, Rosenthal IS, Cavanaugh KC, Pawlak C, Trouille L, et al. (2022) Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas). PLOS ONE 17(1): e0257933. https://doi.org/10.1371/journal.pone.0257933

[4] Finger, D. J. I., McPherson, M. L., Houskeeper, H. F., & Kudela, R. M. (2020, December 17). Mapping Bull kelp canopy in northern California using landsat to enable long-term monitoring. Remote Sensing of Environment. https://www.sciencedirect.com/science/article/pii/S0034425720306167#:~:text=Past%20efforts%20to%20estimate%20kelp,et%20al.%2C%202006 

[5] Mahmood A, Ospina AG, Bennamoun M, An S, Sohel F, Boussaid F, Hovey R, Fisher RB, Kendrick GA. Automatic Hierarchical Classification of Kelps Using Deep Residual Features. Sensors. 2020; 20(2):447. https://doi.org/10.3390/s20020447

[6] Gao, B.-C., Hunt, E. R., Jackson, R. D., Lillesaeter, O., Tucker, C. J., Vane, G., Bowker, D. E., Bowman, W. D., Cibula, W. G., Deering, D., & Elvidge, C. D. (1999, February 22). Ndwi-a normalized difference water index for remote sensing of vegetation liquid water from space. Remote Sensing of Environment. https://www.sciencedirect.com/science/article/abs/pii/S0034425796000673 

[7] GISGeography. (2024, March 10). What is NDVI (normalized difference vegetation index)?. GIS Geography. https://gisgeography.com/ndvi-normalized-difference-vegetation-index/ 

[8] U.S. Department of the Interior. (n.d.). Kelp forests. National Parks Service. https://www.nps.gov/glba/learn/nature/kelp-forest.htm#:~:text=Kelp%20might%20look%20like%20a,Kelp%20does%20not%20have%20roots. 
