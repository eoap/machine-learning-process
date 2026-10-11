config_dict = {
    "output_root": "output",
    "data_ingestion": {
        "root_dir": "output/data_ingestion",
        "local_data_file": "output/data_ingestion",
        "DATA_CLASSES": [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ],
    },
    "prepare_base_model": {
        "root_dir": "output/prepare_base_model",
        "model_path": "output/prepare_base_model/base_model.keras",
        "updated_model_path": "output/prepare_base_model/updated_model.keras",
    },
    "training": {
        "trained_model_path": "output/training/",
        "train_data_dir": "output/data_ingestion/train",
        "val_data_dir": "output/data_ingestion/val",
        "all_data_urls": "output/data_ingestion/",
    },
    "evaluation": {
        "data_dir": "output/data_ingestion/test",
        "all_data_urls": "output/data_ingestion/",
        "trained_model_path": "output/training/trained_model.keras",
    },
}
