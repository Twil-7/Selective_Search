# Selective_Search

无监督图像聚类分割算法：Selective_Search，是RCNN、Fast RCNN的前半部分

环境配置：skimage、opencv

Selective Search algorithm principle：

step 1: calculate the first fel_segment region

step 2: calculate the neighbour couple

step 3: calculate the similarity dictionary

step 4: merge regions and calculate the second merged region

step 5: obtain e target candidate regions by secondary screening
