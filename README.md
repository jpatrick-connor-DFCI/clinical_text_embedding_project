Clinical Text Project:

Code directory overview:
-jupyter notebooks
- python_scripts:
    - data_preprocessing
        - text_preprocessing_and_tokenization.py
            - Extract clinical text and metadata, batch the text, and tokenize
        - generate_clinical_embeddings.py
            - Generate embeddings in batches using the ClinicalLongformer model
        - knit_longformer_embeddings.py
            - Compile the generated embeddings and knit together a metadata dataframe and an embeddings
                numpy array file to use for predictions
        - extract_ICD_times.py
            - Extract all ICDs and their timestamps for every patient in the cohort
        - ICD_to_phecode_map.py
            - Find the corresponding ICD codes in the DFCI system to map to phecodes
        - generate_time-to-phecode_dataset.py
            - For each patient, generate time from first treatment start to death, metastatic disease, and phecodes
        - create_embedding_prediction_dataset.py
            - Add time-decayed embeddings to the time-to-event dataframe to create a prediction dataset
    - model_training
        - run_phecode_predictions_full_cohort.py
            - Train the model across all available patients
        - run_phecode_predictions_stage_subset.py
            - Train the model on the subset of patients with stage, genomics, and text data
    - model_evaluation
        - within_vs_pan_cancer_models.py
            - Compare a model trained within each cancer type with a model trained across all types
    - mortality_trajectories
        - generate_mortality_trajectories.py
            - Perform predictions of mortality risk using additional data to create a mortality curve as a function of treatment time
        - cluster_mortality_trajectories.py
            - Cluster and plot the predicted mortality curves
    - NEPC_prediction
        - compile_prostate_data.py
            - Extract all labs, medications, health history, and text information for prostate cancer patients
        - preprocess_NEPC_prediction_data.py
            - Create a time-to-NEPC dataframe from the first through the tenth PSA test
        - clinical_labs_CoxPH_for_NEPC.py
            - Predict time-to-NEPC using either mean or sequential PSA values
        - text_CoxPH_for_NEPC.py
            - Predict time-to-NEPC using text, text plus mean PSA values, and text plus sequential PSA values
- python_utils:
    - embed_surv_utils:
        - preprocessing.py
            - Contains all helper functions needed to run all scripts in python_scripts/data_preprocessing
        - cox_models.py
            - Contains all functions needed to train and evaluate CoxPH models for phecodes and NEPC time-to-event predictions

