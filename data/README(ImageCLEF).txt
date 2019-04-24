2019 ImageCLEFcoral collection
Jon Chamberlain, Adrian Clark, Antonio Campello and Alba García Seco de Herrera 

Content of files
-----------------
* imageCLEFcoral2019_training.zip
   Development set for both subtasks. 
It contains 240 images with 6702 substrates annotated. Substrates of the same type can have very different morphologies, colour variation and patterns. Some of the images contain a white line (scientific measurement tape) that may occlude part of the entity. The quality of the images is variable, some are blurry, and some have poor colour balance. This is representative of the Marine Technology Research Unit dataset and all images are useful for data analysis.
      
*imageCLEFcoral2019_annotations_training_task_1
 Development set ground truth localised annotations for sub task 1.
The format for the development set of annotated bounding boxes of the substrates is

[image_ID] [seq][substrate] [confidence] [xmin] [ymin] [xmax] [ymax]

 The development set contains 240 images with 6702 substrates annotated. The images have been carefully manual annotated, however it can result in missed annotations or in multiple entities per substrate and there can be multiple substrates per entity. Approximately 10px around the image border is not labelled for technical reasons.

*imageCLEFcoral2019_annotations_training_task_2

 Development set ground truth localised annotations for sub task 2.
The format for the development set of annotated polgons of the substrates is

[image_ID] [seq][sustrate] [confidence] [x1] [y1] [x2] [y2] …[xn] [yn]

 The development set contains 240 images with 6702 substrates annotated. The images have been carefully manual annotated, however it can result in missed annotations or in multiple entities per substrate and there can be multiple substrates per entity. Approximately 10px around the image border is not labelled for technical reasons.





   
   
