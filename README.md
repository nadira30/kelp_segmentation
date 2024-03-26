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

Bla bla bla
Existing methods, like citizen science classification, are labor-intensive and slow, requiring a more efficient approach 
for comprehensive global monitoring. 

### Methods/Approach: Indicate algorithms, methodologies, or approaches you used to craft your solution. What was the reasoning or intuition for trying each methodology/algorithm. What does the overall pipeline look like and the details behindeach component? Make sure to establish any terminology or notation you will continue touse in this section. Note: Your methods and approaches may change through development, so in your project update, feel free to discuss all approaches you tried out! We expect at least 1 method/approach attempted.

Method 1: Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
- Upon inspecting the plots of multiple channel separately, as shown in the image below, we can see that the NIR shows the clearest pattern of the kelp canopies. Therefore, we decided to try using the image of this channel to train, validate, and test our CNN model. 

### Experiments / Results: Describe what you tried and what datasets were used. We aren’t expecting you to beat state of the art, but we are interested in you describing what worked or didn’t work and to give reasoning as to why you believe so. Compare your approach against baselines (either previously established or you established) in this section. Provide at least one qualitative result (i.e. a visual output of your system on an example image). Note: For the project update, feel free to discuss what worked and didn’t work. Why do you think an approach was (un)successful? We expect you to have dealt with dataset setup and completed at least 1 experimental result by the project update.

Method1:  Utilizing the Near-Infrared (NIR) channel of the image with Convolutional Neural Network (CNN).
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

### What’s next: What is your plan until the final project due date? What methods and experiments do you plan on running? Note: Include a task list (can use a table) indicating each step you are planning and anticipated completion date.

### Team member contributions: Indicate what you anticipate each team member will contribute by the final project submission. Note: List every member name and their corresponding tasks in bullet points – or you may simply assign team member names to the task list you created above.

### References
[1] Kelp Forest. Kelp Forest | NOAA Office of National Marine Sanctuaries. (n.d.). https://sanctuaries.noaa.gov/visit/ecosystems/kelpdesc.html 

[2] Browning, J., & Lyons, G. (2020, May 27). 5 reasons to protect kelp, the West Coast’s powerhouse Marine Algae. The Pew Charitable Trusts. https://www.pewtrusts.org/en/research-and-analysis/articles/2020/05/27/5-reasons-to-protect-kelp-the-west-coasts-powerhouse-marine-algae#:~:text=3.-,Protect%20the%20shoreline,filter%20pollutants%20from%20the%20water. 

