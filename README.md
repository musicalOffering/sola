"Soft-Landing Strategy for Alleviating the Task Discrepancy Problem in Temporal Action Localization Tasks"
=============
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Soft-Landing_Strategy_for_Alleviating_the_Task_Discrepancy_Problem_in_Temporal_CVPR_2023_paper.pdf)


Training and assessing the model with the linear evaluation.
=============
1. Prepare the data in the dataset folder. In this code, we will work with the [HACS](https://arxiv.org/pdf/1712.09374.pdf) dataset as the official I3D feature is available [here](http://hacs.csail.mit.edu/hacs_segments_features.zip).
Extract the downloaded feature into the `HACS_DATA` folder.
The file structure should be as follows:   
    ```
    HACS_DATA
    |--training
    |  |--__9PFRfSjE0.npy
    |  |--__9Ux_pexqs.npy
    |  |--...
    |  |--...
    |  ...
    |--validation
    |  |--_-lqeH_0xXU.npy
    |  |--_0EG7L9nCgc.npy
    |  |--...
    |  |--...
    |  ...
    |--training_duration.pkl
    |--training_subset_list.pkl
    |--validation_duration.pkl
    |--validation_subset_list.pkl

    ```    
2. Make `visualizing` directory here and select any 6 `.npy` files from `./HACS_DATA/validation/`. Place them in `./visualizing/`. This is for visualizing the TSM (Temporal Self-similarity Matrix) during the training procedure.

3. Download [evaluation.tar.gz](https://drive.google.com/file/d/1I_e4dIGmWTNBTEYEBbo6uWabbsvsSMn_/view?usp=sharing) and extract it into your working directory so that the file tree looks like this:
    ```
    evalutaion
    |--training_label
    |  |--__9PFRfSjE0.npy
    |  |--__9Ux_pexqs.npy
    |  |--...
    |  |--...
    |  ...
    |--validation_label
    |  |--_-lqeH_0xXU.npy
    |  |--_0EG7L9nCgc.npy
    |  |--...
    |  |--...


    ```   
    This is for the linear evaluation of action/non-action snippet.
4. Run following command:
    ```
    python main_simsiam.py --yaml_path=yamls/hacs_canonical.yaml
    ```
    The code performs both training and linear evaluation on action/non-action snippet features at every `saving_epoch`, as defined in `yamls/hacs_canonical.yaml`. Expect to observe consistent improvements in the evaluation results throughout the training procedure.
    
5. After training, you can choose one of the saved model (which resulted in the best linear evaluation result) and run following
    ```
    python process_feature.py --yaml_path=yamls/hacs_canonicla.yaml --load_model=150
    ```
    to generate SoLa features.
    With SoLa features, you can get better result in [G-TAD](https://arxiv.org/pdf/1911.11462.pdf) downstream head.