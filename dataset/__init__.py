from dataset.Custom import create_custom_dataset

def create_dataset(dataset: str, **kwargs):
    if dataset == "custom":
        return create_custom_dataset(**kwargs)
    else:
        raise ValueError(f"dataset except {'custom'}, but got {dataset}")
