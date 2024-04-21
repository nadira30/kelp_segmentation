# Kelp Segmentation: Mapping and Monitoring of Kelp Forests
### Team Member: 
* Nadira Amadou
* Oluwatofunmi Sodimu 
* Thanapol Tantagunninat

### Introduction/Problem Definition:
---------------------------------------------------------------------
Introduction (30pt):
○ (10pt) High Level Description and Motivation: What is your high-level topic
area? Why should people care about this project? If you succeed, what benefit
will you provide and to whom?
○ (10pt) Specific Problem Definition: What precise problem are you tackling? What
is the desired goal? What is the expected input and output?
○ (10pt) Visuals: Include at least one diagram/visual to help describe your problem
statement and/or goal. This can be an example input/output, an example of current
failures of existing methods, or a diagram explaining your motivation
-------------------------------------------------------------------------

A kelp forest is an ocean community made up of dense groupings of kelps. These forests typically serve as a source of food, shelter, and protection to a wide variety of marine life [1]. Moreover, the significance of kelp forests extends beyond their role as a biodiverse hotspot. They play a pivotal role in regulating marine environments and supporting human well-being. Kelps are renowned for their capacity to sequester carbon dioxide through photosynthesis, contributing significantly to carbon capture from human's daily life and carbon storage in the oceans [2]. Additionally, these underwater havens act as natural buffers against coastal erosion, mitigating the impacts of waves and currents on shorelines. Furthermore, kelp forests hold economic value, serving as crucial fishing grounds and tourist attractions in many coastal regions.

However, these forests face threats from multitude of factors such as climate change, acidification, overfishing, and unsustainable harvesting practices [1]. 

To address these pressing conservation concerns and safeguard the future of kelp forests, innovative approaches are needed. One promising strategy involves harnessing the power of technology to monitor and protect these vital marine habitats. By leveraging advances in remote sensing, computer vision, and machine learning, we propose the development of a comprehensive model capable of monitoring kelp forests using coastal satellite imagery. Such a model would enable tracking of changes in kelp abundance, empowering conservation efforts and informing sustainable management practices.

| ![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/965b146e-3d3b-4394-955c-e50391a7ab1c) | 
|:--:| 
| *Image Showing Sardines Seeking Food and Shelter in a Kelp Forest [2]*|

![kelpswir_oli_2019](https://github.com/nadira30/kelp_segmentation/assets/128086407/7d7aa4ab-a17f-45c3-815f-8f9814d8d7c5)
|:--:| 
| *Image Showing the change in Kelp forest abundance from 2008 to 2019* |

We aim to develop a segmentation model based of a Convolutional Neural Network (CNN). The input to our model are the satellite images of 350x350 pixels containing 7 channels of which we will experiment with different selection/combination of channels. The output/results is the predicted labels which represents the area where it contains kelp (1) or no kelp (0). The goal is to successfully match the predicted kelp labels with the ground truth label in the dataset we used to validate our performance.

![7channel](https://github.com/nadira30/kelp_segmentation/assets/128086407/18671950-b74d-4559-8287-25e96f84c3c4)
|:--:| 
| *Image Showing the 7 channels in the satellite imagery* |


|:--:| 
| *Image Showing the Ground truth label of pixels containing kelp as the expected output* |


### Related Works: 

--------------------------------------------------------------------------
Related Work (30pt):
○ (20pt) Explaining Context: Clearly state what other people have done before to
solve your specific problem. If your project brings together multiple areas, you
will need to describe each area and what people have done in each.
■ Provides citations of the five most relevant references to your work.
Please cite works following IEEE guidelines.
■ For each citation, provides a short (2-6 sentence) explanation as to
• Why this work is relevant to your project.
• What this work contributed to the field.
○ (10pt) Your project in context: What was missing in prior work and how does
your work fill this hole?
--------------------------------------------------------------------------

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

--------------------------------------------------------------------------
Method/Approach (50pt):
○ (20pt) Method Overview:
■ Clearly specify your current best approach.
■ Define all necessary notation / terminology.
■ Specify each component of your approach and for non-standard
components describe the details that would be necessary for someone
familiar with computer vision to understand what you are doing.
• E.g. If you are building off an architecture from prior works,
please briefly describe the architecture.
○ (10pt) Contribution:
■ Why do you expect your approach to work better than what people have
done before? What does your work contribute?
• E.g., If you are building a new method on top of prior work, what
component is new? If you are re-using methods on new application
areas, clearly specify that no one has tried these approaches for
this application.
○ (10pt) Intuition:
■ Explain why you expect your approach to work to solve the limitation of
the literature described in your related work.
○ (10pt) Visuals:
■ Provide a diagram that helps to explain your approach. This diagram could
describe your approach pipeline, a key component you are introducing, or
another visual that you feel helps best explain what you need to convey
your approach.
--------------------------------------------------------------------------
### Experiments / Results: 

--------------------------------------------------------------------------
We arrived at the approach discussed in the Methods section through the following experiments:
Model Architecture: We experimented with different neural networks for semantic segmentation. 
Model Parameters: We experimented with different loss functions e.g. binary cross-entropy and our custom dice_loss function to train our neural network. We also tried out different values for the learning rate, optimizer, model depth, etc.
Data Preprocessing: We experimented with different data pre-processing methods such as the use of the Normalized Difference Water Index (NDWI) or the Normalized Difference Vegetation Index (NDVI) parameters which are both used for determining the presence/absence of vegetation. The most appropriate data pre-processing techniques will be chosen based on the accuracy values recorded when the processed data is used to train our model.

By optimizing the model architecture and parameters, as well as the data preprocessing methods, we should arrive at a model that accurately maps kelp in satellite images.

To implement our method for kelp segmentation of satellite images, we made use of the dataset from the ‘Kelp Wanted: Segmenting Kelp Forests’ competition on the driven data website [10]. The training set includes 5,635 TIFF images/label pairs, and the test set includes 1,426 TIFF images. Each image/label has a size of 350x350 pixels, with the input image having 7 channels and the label having 1 channel (binary mask of kelp or no kelp). The 7 channels of the input image are described below. 

Short-Wave Infrared (SWIR): The SWIR band is highly useful for differentiating between water and land. Water bodies absorb more SWIR light, appearing darker, which can help distinguish kelp, which might reflect slightly more SWIR than plain open water.
Near-Infrared (NIR): NIR is generally absorbed by water but reflected by vegetation. Kelp forests will likely show higher reflectance in this band compared to the surrounding water, making NIR a critical channel for identifying aquatic vegetation.
 Red (R): The red band is absorbed by chlorophyll in healthy vegetation, making it less reflective in vegetation-rich areas but more reflective in areas without vegetation. However, underwater, the red light is absorbed quickly, which might reduce its utility unless the water is very shallow.
 Green (G): This band might be helpful in mapping shallow underwater habitats since it can penetrate deeper than the red channel. The green color also gets absorbed by the natural green color of the kelp.
Blue (B): Blue light penetrates deeper into the water than other visible wavelengths, which can help in identifying deeper kelp forests.
Cloud Mask: The cloud mask is useful for preprocessing to eliminate areas obscured by clouds from your analysis, ensuring that only clear observations of the water and kelp are considered.
Elevation: An elevation map (ground mask) serves as a ground mask by identifying land areas. We can utilize this map to exclude these land areas from the regions considered for kelp presence, ensuring more accurate prediction results.

Due to the unavailability of the ground truth labels for the test set, we split the training set by a ratio of 70-15-15 for training, validating, and testing our model. Sample images of the RGB channels, NIR, SWIR, Cloud Mask, Elevation map, and label are shown below:

|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/06ec8f64-cd38-4568-83ef-b3c24c4f0d2c)|
|:--:| 
| SWIR Channel |
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/4449d77b-cdeb-4d47-8d92-fa3670435d9a)|
|:--:| 
| NIR Channel |
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/70aa5c40-e1b5-46b1-97ab-54430ce5a6f2)|
|:--:| 
| RGB Channels |
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/774604e5-5993-48de-8d04-e128b86ef754)|
|:--:| 
| Cloud Mask |
|![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/a475e6d9-a005-47fc-a80f-befb0078deb8)|
|:--:| 
| Elevation Map |
![image](https://github.com/nadira30/kelp_segmentation/assets/35805326/c6149804-e47e-4f7b-9f7f-2fc7045f7175)|
|:--:| 
| Kelp Label |


The desired output of our model is a binary mask/image with 0 representing the absence of kelp and 1 representing the presence of kelp. A sample output can be seen in the picture titled ‘Kelp Label’.

For evaluating our approach, we made use of the dice coefficient and mean Intersection over Union metrics as these work well with highly class imbalanced datasets as is common in semantic segmentation tasks.


Results (35pt)
○ (10pt) Baselines
■ How do prior works perform on this task? It’s best to have a quantitative
comparison using your metric for success. If that is not possible, a
qualitative result will suffice.
■ You are required to have at least one comparison.
○ (10pt) Key result presentation
■ Clearly present your key result. This should include both your
performance according to your metric of success defined above and a
qualitative output example from your system.
○ (15pt) Key result performance
■ Specify what variants of your approach you tried to make progress
towards finding a solution.
■ Ultimately, describe your final results. Did your approach work? If not,
explain why you believe it did not.
--------------------------------------------------------------------------

### Discussion
--------------------------------------------------------------------------
Discussion (15pt)
○ Provide a reflective paragraph about your project. Summarize what you
accomplished, what you learned, and what remains for future works.
○ If you were to start over today, is that a different strategy you would take to make
progress on your overall problem statement?
--------------------------------------------------------------------------

  
### Challenges Encountered

--------------------------------------------------------------------------
● Challenges Encountered (10pt)
○ What challenges did your team encounter throughout the project?
--------------------------------------------------------------------------



### Team member contributions: 

--------------------------------------------------------------------------
● Team Member Contributions (10pt)
○ Each team member should individually fill out the team member contribution MS
form. https://forms.office.com/r/9HSrYJNvik (you will need to login to your GT
credentials to complete this form, one response per student).
--------------------------------------------------------------------------

Nadira Amadou:
- Formatting dataset 
- Develop a custom CNN architecture
- Training and testing method 3

Oluwatofunmi Sodimu:
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

[9] NASA Earth Observatory. Monitoring the Collapse of Kelp Forests. https://earthobservatory.nasa.gov/images/148391/monitoring-the-collapse-of-kelp-forests
