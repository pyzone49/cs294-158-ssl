import matplotlib.pyplot as plt
from deepul_helper.demos import load_model_and_data


if __name__ == '__main__':
    # Load model and data
    model, linear_classifier, train_loader, test_loader = load_model_and_data('puzzle',dataset="pascalvoc2012")

    # Get a single batch from the loader
    data_iter = iter(train_loader)
    batch_uniform_patches, batch_random_patches, batch_labels = next(data_iter)  # Assuming your dataset __getitem__ returns these three