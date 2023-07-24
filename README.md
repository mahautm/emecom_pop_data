Code to reproduce datasets for emergent communication experiments

Preresequite : 
    - For the imagenet validation split, ILSVRC02012 validation data must be downloaded,
    as well as the annotation file containing the labels from the devkit
    - For the out of domain subset, preprocessed imagenet21k must be downloaded
both are available  on the official imagenet website : https://image-net.org/download-images.php

Scripts : 
    * reproduce_datasets.py
    
        * Start here. Uses a pytorch dataloader to access data from either the split of imagenet1k validation data used in the experiments
        or the out of domain data from the imagenet21k selected classes
        
    * data_utils.py
    
        * functions and classes to prepare and organise data
        
    * imood_class_ids.npy
    
        * numpy array containing the ids of all selected out of domain classes
