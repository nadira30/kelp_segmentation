# Kelp Segmentation: Mapping and Monitoring of Kelp Forests
Computer Vision Class project: https://www.drivendata.org/competitions/255/kelp-forest-segmentation/data/

### Introduction/Problem Definition:

A kelp forest is an ocean community made up of dense groupings of kelps. These forests typically serve as a source of food, shelter, and protection to a wide variety of marine life [1]. Moreover, the significance of kelp forests extends beyond their role as a biodiverse hotspot. They play a pivotal role in regulating marine environments and supporting human well-being. Kelps are renowned for their capacity to sequester carbon dioxide through photosynthesis, contributing significantly to carbon capture from human's daily life and carbon storage in the oceans [2]. Additionally, these underwater havens act as natural buffers against coastal erosion, mitigating the impacts of waves and currents on shorelines. Furthermore, kelp forests hold economic value, serving as crucial fishing grounds and tourist attractions in many coastal regions.

However, these forests face threats from multitude of factors such as climate change, acidification, overfishing, and unsustainable harvesting practices [1]. 

To address these pressing conservation concerns and safeguard the future of kelp forests, innovative approaches are needed. One promising strategy involves harnessing the power of technology to monitor and protect these vital marine habitats. By leveraging advances in remote sensing, computer vision, and machine learning, we propose the development of a comprehensive model capable of monitoring kelp forests using coastal satellite imagery. Such a model would enable tracking of changes in kelp abundance, empowering conservation efforts and informing sustainable management practices.

| ![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/965b146e-3d3b-4394-955c-e50391a7ab1c) | 
|:--:| 
| Image Showing Sardines Seeking Food and Shelter in a Kelp Forest [2] |


![kelpswir_oli_2019](https://github.com/nadira30/kelp_segmentation/assets/128086407/7d7aa4ab-a17f-45c3-815f-8f9814d8d7c5)
|:--:| 
| Image Showing the change in Kelp forest abundance from 2008 to 2019 |

### Related Works: Describe related works in your problem space (research papers,libraries/tools, etc) for existing solutions for this problem or adjacent areas. Make sure to cite papers you reference! Note: Related work should have between 2 to 6 sentences for each work citing. Please cite works following IEEE guidelines. Organize related work into different subsections based on similarity of approaches.

#### [3] Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas):
- The paper evaluates two methods to automate kelp detection from satellite imagery: 1) crowdsourced classifications from the Floating Forests citizen science project (FF8), and 2) an automated spectral approach using a decision tree combined with multiple endmember spectral mixture analysis (DTM).
- Both methods were applied to classify kelp from Landsat 5, 7, 8 imagery covering the Falkland Islands from 1985-2021.
- DTM showed better performance than FF8 when validated against expert manual classifications of 8 Landsat scenes covering over 2,700 km of coastline.
- Multiple Endmember Spectral Mixture Analysis (MESMA) is a spectral unmixing algorithm that estimates the fractional contributions of pure spectral endmembers (e.g. water, kelp, glint) to each image pixel spectrum based on a linear mixing model. It allows estimating partial kelp coverage within pixels.
- Decision Tree Classification is used to first identify potential candidate kelp-containing pixels before applying MESMA. The decision tree uses spectral rules to separate kelp from non-kelp pixels. Then MESMA spectral unmixing is utilized to estimate fractional kelp coverage within those candidate pixels
  
![results_remote_sensing](https://github.com/nadira30/kelp_segmentation/assets/128086407/bd4350ca-dd28-45f5-9a77-abd99fae646c)
|:--:| 
| Image Showing the results of automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas)|

#### [4] Mapping bull kelp canopy in northern California using Landsat to enable long-term monitoring
- This paper is focused on the mapping and monitoring of kelp, specifically bull kelp, though the use of Landsat satellite images.
- Compared to aerial imagery and private satellite images that have been employed by other researchers, Landsat images allow for long-term montioring since they are updated by the U.S. Geological Survey (USGS) every 16 days.

### Methods/Approach: Indicate algorithms, methodologies, or approaches you used to craft your solution. What was the reasoning or intuition for trying each methodology/algorithm. What does the overall pipeline look like and the details behindeach component? Make sure to establish any terminology or notation you will continue touse in this section. Note: Your methods and approaches may change through development, so in your project update, feel free to discuss all approaches you tried out! We expect at least 1 method/approach attempted.

#### Method 1: Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
- Upon inspecting the plots of multiple channel separately, as shown in the image below, we can see that the NIR shows the clearest pattern of the kelp canopies. Therefore, we decided to try using the image of this channel to train, validate, and test our CNN model.
  
#### Method 2: Utilizing the Normalized Difference Water Index (NDWI), Normalized Difference Vegetation Index (NDVI), and RGB channels with the U-Net Convolutional Neural Network Architecture.
- The Normalized Difference Water Index (NDWI) is a parameter that may be used to differentiate between different types of vegetation. NDWI = (Near Infrared - Shortwave Infrared)/(Near Infrared + Shortwave Infrared) [6]. Typically, values between -1 and 0 indicate a lack of vegetation or water content, while values greater than 1 indicate the presence of water [6].
- The Normalized Difference Vegetation Index (NDVI) is a parameter that may also be used to differentiate between different types of vegetation. NDVI = (Near Infrared  - Red)/(Near Infrared + Red) [7]. NDVI values typically fall within the range of -1 and +1 with the value increasing in proportion to the increase in vegetation [7]. An NDVI of 0 may indicate a lack of vegetation (e.g. buildings), an NDVI of -1 may indicate a large body of water, and an NDVI of +1 may indicate dense vegetation [7].
- Due to the utility of these parameters in detecting the presence of vegetation, they were used in combination with the RGB channels  to train a U-Net model that would be able to return a semantically segmented image with labels corresponding to kelp(1) or no kelp(0).
- Additionally, the digital elevation map values and the cloud mask values were used to filter out irreleveant pixels prior to training.
  
### Experiments / Results: Describe what you tried and what datasets were used. We aren’t expecting you to beat state of the art, but we are interested in you describing what worked or didn’t work and to give reasoning as to why you believe so. Compare your approach against baselines (either previously established or you established) in this section. Provide at least one qualitative result (i.e. a visual output of your system on an example image). Note: For the project update, feel free to discuss what worked and didn’t work. Why do you think an approach was (un)successful? We expect you to have dealt with dataset setup and completed at least 1 experimental result by the project update.

#### Method1:  Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
- The dataset was splitted into train-val-test ratio of 70-15-15. The CNN's architecture comprises:
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


The visual results and quantitative results are shown in the image below:
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/625de30b-2a63-46a3-9cf5-fd90cfff8049)|
|:--:| 
| Image Showing Results of Method 2 |

### What’s next: What is your plan until the final project due date? What methods and experiments do you plan on running? Note: Include a task list (can use a table) indicating each step you are planning and anticipated completion date.

### Team member contributions: Indicate what you anticipate each team member will contribute by the final project submission. Note: List every member name and their corresponding tasks in bullet points – or you may simply assign team member names to the task list you created above.

### References
[1] Kelp Forest. Kelp Forest | NOAA Office of National Marine Sanctuaries. (n.d.). https://sanctuaries.noaa.gov/visit/ecosystems/kelpdesc.html 

[2] Browning, J., & Lyons, G. (2020, May 27). 5 reasons to protect kelp, the West Coast’s powerhouse Marine Algae. The Pew Charitable Trusts. https://www.pewtrusts.org/en/research-and-analysis/articles/2020/05/27/5-reasons-to-protect-kelp-the-west-coasts-powerhouse-marine-algae#:~:text=3.-,Protect%20the%20shoreline,filter%20pollutants%20from%20the%20water. 

[3] Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas)
Houskeeper HF, Rosenthal IS, Cavanaugh KC, Pawlak C, Trouille L, et al. (2022) Automated satellite remote sensing of giant kelp at the Falkland Islands (Islas Malvinas). PLOS ONE 17(1): e0257933. https://doi.org/10.1371/journal.pone.0257933

[4] Finger, D. J. I., McPherson, M. L., Houskeeper, H. F., & Kudela, R. M. (2020, December 17). Mapping Bull kelp canopy in northern California using landsat to enable long-term monitoring. Remote Sensing of Environment. https://www.sciencedirect.com/science/article/pii/S0034425720306167#:~:text=Past%20efforts%20to%20estimate%20kelp,et%20al.%2C%202006 

[6] Gao, B.-C., Hunt, E. R., Jackson, R. D., Lillesaeter, O., Tucker, C. J., Vane, G., Bowker, D. E., Bowman, W. D., Cibula, W. G., Deering, D., & Elvidge, C. D. (1999, February 22). Ndwi-a normalized difference water index for remote sensing of vegetation liquid water from space. Remote Sensing of Environment. https://www.sciencedirect.com/science/article/abs/pii/S0034425796000673 

[7] GISGeography. (2024, March 10). What is NDVI (normalized difference vegetation index)?. GIS Geography. https://gisgeography.com/ndvi-normalized-difference-vegetation-index/ 
