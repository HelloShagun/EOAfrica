import os
import rasterio
import json
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.cluster import HDBSCAN
import numpy as np
import os
import rasterio
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
from my_package.landsat_indices import calculate_indices
from my_package.thresholding_methods import *
from my_package.sentinel2_indices import calculate_indices_S2
from my_package.raster_vector import raster_to_vector
from my_package.data_downloading import download_band, save_all_bands
from my_package.models_thresholding import otsu_threshold, gmm_threshold, adaptive_threshold
from my_package.plotting_and_display import pixel_count_for_area
from my_package.plotting_and_display import compute_contingency_map, display_masks, plot_histogram_with_threshold, generate_mask, compute_segmentation_metrics
import pandas as pd
import rasterio
import math
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt




# Function to load a single band
def load_band(filepath):
    with rasterio.open(filepath) as src:
        return src.read(1), src.profile  # Read the first band
        
def load_mask_and_sample(raster_path, num_samples=50):
    """
    Loads the mask raster and performs random sampling for two classes (0 and 1).

    Args:
        raster_path (str): Path to the mask raster.
        num_samples (int): Number of samples per class.

    Returns:
        tuple: (sampled_coords, sampled_labels) where sampled_coords are pixel locations (row, col)
    """
    with rasterio.open(raster_path) as src:
        mask_data = src.read(1)  # Read first band
        transform = src.transform
    
    # Get water (1) and non-water (0) pixel locations
    water_pixels = np.argwhere(mask_data == 1)
    non_water_pixels = np.argwhere(mask_data == 0)

    # Randomly sample equal numbers from both classes
    sampled_water = water_pixels[np.random.choice(water_pixels.shape[0], num_samples, replace=False)]
    sampled_non_water = non_water_pixels[np.random.choice(non_water_pixels.shape[0], num_samples, replace=False)]

    # Combine both classes
    sampled_coords = np.vstack((sampled_water, sampled_non_water))
    sampled_labels = np.hstack((np.ones(num_samples), np.zeros(num_samples)))  # 1 = Water, 0 = Non-water

    return sampled_coords, sampled_labels, transform



# def load_mask_and_sample(raster_path, sampling_ratio=0.1):
#     """
#     Loads the mask raster and performs random sampling for two classes (0 and 1)
#     by selecting a specified percentage of pixels from each class.

#     Args:
#         raster_path (str): Path to the mask raster.
#         sampling_ratio (float): Percentage of pixels to sample from each class (default is 0.1 = 10%).

#     Returns:
#         tuple: (sampled_coords, sampled_labels, transform) where:
#             - sampled_coords: Pixel locations (row, col)
#             - sampled_labels: Labels corresponding to the sampled pixels (0 = Non-water, 1 = Water)
#             - transform: The transform of the raster file
#     """
#     with rasterio.open(raster_path) as src:
#         mask_data = src.read(1)  # Read first band
#         transform = src.transform
    
#     # Get water (1) and non-water (0) pixel locations
#     water_pixels = np.argwhere(mask_data == 1)
#     non_water_pixels = np.argwhere(mask_data == 0)
    
#     # Determine number of samples based on sampling ratio
#     num_samples_water = max(1, int(len(water_pixels) * sampling_ratio))
#     num_samples_non_water = max(1, int(len(non_water_pixels) * sampling_ratio))

#     # If there's not enough pixels, adjust the number of samples
#     if num_samples_water > len(water_pixels):
#         num_samples_water = len(water_pixels)
        
#     if num_samples_non_water > len(non_water_pixels):
#         num_samples_non_water = len(non_water_pixels)

#     # Randomly sample pixels from each class
#     sampled_water = water_pixels[np.random.choice(water_pixels.shape[0], num_samples_water, replace=False)]
#     sampled_non_water = non_water_pixels[np.random.choice(non_water_pixels.shape[0], num_samples_non_water, replace=False)]

#     # Combine both classes
#     sampled_coords = np.vstack((sampled_water, sampled_non_water))
#     sampled_labels = np.hstack((np.ones(len(sampled_water)), np.zeros(len(sampled_non_water))))  # 1 = Water, 0 = Non-water

#     return sampled_coords, sampled_labels, transform


def extract_features_from_bands(loaded_bands, sampled_coords):
    """
    Extracts feature values from loaded bands at sampled coordinates.

    Args:
        loaded_bands (dict): Dictionary containing raster layers (water indices).
        sampled_coords (np.array): Array of (row, col) coordinates.

    Returns:
        dict: Extracted feature data per index name.
    """
    feature_data = {index: [] for index in loaded_bands.keys()}  # Store extracted values

    for index, band_data in loaded_bands.items():
        for row, col in sampled_coords:
            feature_data[index].append(band_data[row, col])  # Extract pixel value

    # Convert to NumPy arrays for training
    for index in feature_data:
        feature_data[index] = np.array(feature_data[index])

    return feature_data


def train_rf_models(features_per_index, sampled_labels):
    """
    Trains one RF model per index and one RF model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.
        sampled_labels (np.array): Array of labels (0 = Non-water, 1 = Water).

    Returns:
        dict: Trained RF models (one per index + combined).
    """
    models = {}

    # Train one model per index
    # for index, features in features_per_index.items():
    #     rf = RandomForestClassifier(n_estimators=100, random_state=42)
    #     rf.fit(features.reshape(-1, 1), sampled_labels)
    #     models[index] = rf

    # Train RF with all indices combined
    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])

    num_pixels = combined_features.shape[0]
    print(f"Number of pixels used for training: {num_pixels}")
    
    rf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_combined.fit(combined_features, sampled_labels)
    models["Combined"] = rf_combined

    return models

def classify_full_raster_rf(loaded_bands, models):
    """
    Applies trained RF models to classify the entire raster.

    Args:
        loaded_bands (dict): Dictionary of raster layers (water indices).
        models (dict): Dictionary of trained RF models.

    Returns:
        dict: Classified raster results per model.
    """
    classified_rasters = {}

    # Reshape raster data for prediction
    raster_shape = next(iter(loaded_bands.values())).shape
    num_pixels = raster_shape[0] * raster_shape[1]

    # Print the number of pixels used for inference
    print(f"Number of pixels used for inference: {num_pixels}")

    for index, model in models.items():
        if index == "Combined":
            # Predict using all bands combined
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
        else:
            # Predict using a single band
            raster_features = loaded_bands[index].ravel().reshape(-1, 1)

        predictions = model.predict(raster_features)  # Predict classes
        classified_rasters[index] = predictions.reshape(raster_shape)  # Reshape back to raster shape

    return classified_rasters

import numpy as np
import matplotlib.pyplot as plt
import math

def compute_contingency_map_2(classified_rasters, reference_mask):
    """
    Generates and displays contingency maps for each classified raster using automatic subplots.
    Color Coding:
        - True Positive (TP): Blue
        - False Positive (FP): Red
        - False Negative (FN): Black
        - True Negative (TN): White

    Args:
        classified_rasters (dict): Dictionary of classified raster results.
        reference_mask (np.array): Ground truth binary mask.
    """
    num_maps = len(classified_rasters)
    
    # Automatically determine the grid size (rows, cols) for better visualization
    cols = math.ceil(math.sqrt(num_maps))
    rows = math.ceil(num_maps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    
    # If there's only one plot, axes is not an array
    if num_maps == 1:
        axes = [axes]

    # Flatten the axes array for easy iteration if it's a 2D array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for i, (index, classified_raster) in enumerate(classified_rasters.items()):
        ax = axes[i]
        
        # Ensure both masks are binary (0 or 1)
        predicted_mask = (classified_raster > 0).astype(int)
        reference_mask_binary = (reference_mask > 0).astype(int)

        # Create an empty RGB image
        contingency_map = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

        # True Positive (TP): Blue
        tp = (predicted_mask == 1) & (reference_mask_binary == 1)
        contingency_map[tp] = [0, 0, 255]

        # False Positive (FP): Red
        fp = (predicted_mask == 1) & (reference_mask_binary == 0)
        contingency_map[fp] = [255, 0, 0]

        # False Negative (FN): Black
        fn = (predicted_mask == 0) & (reference_mask_binary == 1)
        contingency_map[fn] = [0, 0, 0]

        # True Negative (TN): White
        tn = (predicted_mask == 0) & (reference_mask_binary == 0)
        contingency_map[tn] = [255, 255, 255]
        
        # Plot the contingency map
        ax.imshow(contingency_map)
        ax.set_title(f"{index}", fontsize=15)
        ax.axis("off")
    
    # Hide any extra subplots if they exist
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the figure to the current folder
    figure_name = f"contingency"
    plt.savefig(figure_name)
    #plt.close(fig)
        
    print(f"Figure saved to: {figure_name}")
    
    plt.show()




def train_gmm_models(features_per_index, sampled_labels):
    """
    Trains one GMM model per index and one GMM model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.
        sampled_labels (np.array): Array of labels (0 = Non-water, 1 = Water).

    Returns:
        dict: Trained GMM models (one per index + combined).
    """
    models = {}

    # for index, features in features_per_index.items():
    #     gmm = GaussianMixture(n_components=2, random_state=42)
    #     gmm.fit(features.reshape(-1, 1))
    #     models[index] = gmm

    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])
    gmm_combined = GaussianMixture(n_components=2, random_state=42)
    gmm_combined.fit(combined_features)
    models['Combined'] = gmm_combined

    return models


def train_svm_models(features_per_index, sampled_labels):
    """
    Trains one SVM model per index and one SVM model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.
        sampled_labels (np.array): Array of labels (0 = Non-water, 1 = Water).

    Returns:
        dict: Trained SVM models (one per index + combined).
    """
    models = {}

    # for index, features in features_per_index.items():
    #     svm = SVC(probability=True, random_state=42)
    #     svm.fit(features.reshape(-1, 1), sampled_labels)
    #     models[index] = svm

    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])

    num_pixels = combined_features.shape[0]
    print(f"Number of pixels used for training: {num_pixels}")
    
    svm_combined = SVC(probability=True, random_state=42)
    svm_combined.fit(combined_features, sampled_labels)
    models['Combined'] = svm_combined

    return models


def train_hdbscan_models(features_per_index):
    """
    Trains one HDBSCAN model per index and one HDBSCAN model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.

    Returns:
        dict: Trained HDBSCAN models (one per index + combined).
    """
    models = {}

    # for index, features in features_per_index.items():
    #     hdbscan_model = HDBSCAN(min_cluster_size=10)
    #     hdbscan_model.fit(features.reshape(-1, 1))
    #     models[index] = hdbscan_model

    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])
    hdbscan_combined = HDBSCAN(min_cluster_size=10)
    hdbscan_combined.fit(combined_features)
    models['Combined'] = hdbscan_combined

    return models


def classify_full_raster_gmm(loaded_bands, models):
    """
    Applies trained GMM models to classify the entire raster.
    """
    classified_rasters = {}
    raster_shape = next(iter(loaded_bands.values())).shape

    for index, model in models.items():
        if index == 'Combined':
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
        else:
            raster_features = loaded_bands[index].ravel().reshape(-1, 1)

        predictions = model.predict(raster_features)
        classified_rasters[index] = predictions.reshape(raster_shape)

    return classified_rasters


def classify_full_raster_svm(loaded_bands, models):
    """
    Applies trained SVM models to classify the entire raster.
    """
    classified_rasters = {}
    raster_shape = next(iter(loaded_bands.values())).shape

    for index, model in models.items():
        if index == 'Combined':
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
        else:
            raster_features = loaded_bands[index].ravel().reshape(-1, 1)

        predictions = model.predict(raster_features)
        classified_rasters[index] = predictions.reshape(raster_shape)

    return classified_rasters




def classify_full_raster_hdbscan(loaded_bands, models):
    """
    Applies trained HDBSCAN models to classify the entire raster.
    """
    classified_rasters = {}
    raster_shape = next(iter(loaded_bands.values())).shape

    for index, model in models.items():
        if index == 'Combined':
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
            predictions = model.fit_predict(raster_features)  # Perform prediction
        else:
            # raster_features = loaded_bands[index].ravel().reshape(-1, 1)
            predictions = model.fit_predict(raster_features)  # Perform prediction

        classified_rasters[index] = predictions.reshape(raster_shape)

    return classified_rasters


def train_knn_models(features_per_index, sampled_labels, n_neighbors=5):
    """
    Trains one KNN model per index and one KNN model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.
        sampled_labels (np.array): Array of labels (0 = Non-water, 1 = Water).
        n_neighbors (int): Number of neighbors to use for KNN (default is 5).

    Returns:
        dict: Trained KNN models (one per index + combined).
    """
    models = {}

    # for index, features in features_per_index.items():
    #     knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    #     knn.fit(features.reshape(-1, 1), sampled_labels)
    #     models[index] = knn

    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])

    num_pixels = combined_features.shape[0]
    print(f"Number of pixels used for training: {num_pixels}")
    
    knn_combined = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_combined.fit(combined_features, sampled_labels)
    models['Combined'] = knn_combined

    return models


def train_mlc_models(features_per_index, sampled_labels):
    """
    Trains one MLC model (QDA) per index and one MLC model for all indices combined.

    Args:
        features_per_index (dict): Dictionary containing feature arrays for each index.
        sampled_labels (np.array): Array of labels (0 = Non-water, 1 = Water).

    Returns:
        dict: Trained MLC models (one per index + combined).
    """
    models = {}

    # for index, features in features_per_index.items():
    #     mlc = QuadraticDiscriminantAnalysis()
    #     mlc.fit(features.reshape(-1, 1), sampled_labels)
    #     models[index] = mlc

    combined_features = np.column_stack([features_per_index[idx] for idx in features_per_index.keys()])
    num_pixels = combined_features.shape[0]
    print(f"Number of pixels used for training: {num_pixels}")
    
    mlc_combined = QuadraticDiscriminantAnalysis()
    mlc_combined.fit(combined_features, sampled_labels)
    models['Combined'] = mlc_combined

    return models

def classify_full_raster_knn(loaded_bands, models):
    """
    Applies trained KNN models to classify the entire raster.
    """
    classified_rasters = {}
    raster_shape = next(iter(loaded_bands.values())).shape

    for index, model in models.items():
        if index == 'Combined':
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
        else:
            raster_features = loaded_bands[index].ravel().reshape(-1, 1)

        predictions = model.predict(raster_features)
        classified_rasters[index] = predictions.reshape(raster_shape)

    return classified_rasters


def classify_full_raster_mlc(loaded_bands, models):
    """
    Applies trained MLC (QDA) models to classify the entire raster.
    """
    classified_rasters = {}
    raster_shape = next(iter(loaded_bands.values())).shape

    for index, model in models.items():
        if index == 'Combined':
            raster_features = np.column_stack([loaded_bands[idx].ravel() for idx in loaded_bands.keys()])
        else:
            raster_features = loaded_bands[index].ravel().reshape(-1, 1)

        predictions = model.predict(raster_features)
        classified_rasters[index] = predictions.reshape(raster_shape)

    return classified_rasters



def choose_index_and_extract_features(loaded_bands, index_choice, sampled_coords):
    """
    Allows user to choose which index to work with and extracts features accordingly.

    Args:
        loaded_bands (dict): Loaded raster layers.
        index_choice (list): List of index names to work with (e.g., index_names_1 or index_names_2).
        sampled_coords (np.array): Sampled coordinates for feature extraction.

    Returns:
        dict: Extracted features for the chosen index.
    """
    loaded_bands_index = dictfilt(loaded_bands, index_choice)
    features_per_index = extract_features_from_bands(loaded_bands_index, sampled_coords)
    return loaded_bands_index, features_per_index


def generate_combined_image(loaded_bands_index, features_per_index, sampled_labels, reference_mask, index_type, lake_number):
    """
    Trains models on chosen indices and generates a combined image of results.

    Args:
        loaded_bands_index (dict): Raster layers for selected indices.
        features_per_index (dict): Extracted features for selected indices.
        sampled_labels (np.array): Ground truth labels.
        reference_mask (np.array): Reference mask for comparison.

    Returns:
        None
    """
    # Train Models
    model_trained_rf = train_rf_models(features_per_index, sampled_labels)
    model_trained_svm = train_svm_models(features_per_index, sampled_labels)
    model_trained_knn = train_knn_models(features_per_index, sampled_labels)
    model_trained_mlc = train_mlc_models(features_per_index, sampled_labels)



    # Classify Full Raster
    classified_rasters_rf = classify_full_raster_rf(loaded_bands_index, model_trained_rf)
    classified_rasters_svm = classify_full_raster_svm(loaded_bands_index, model_trained_svm)
    classified_rasters_knn = classify_full_raster_knn(loaded_bands_index, model_trained_knn)
    classified_rasters_mlc = classify_full_raster_mlc(loaded_bands_index, model_trained_mlc)
    
    # Plot Combined Image
    acc_scores = plot_combined_contingency_maps(
        classified_rasters_rf,
        classified_rasters_svm,
        classified_rasters_knn,
        classified_rasters_mlc,
        reference_mask,
        index_type,
        lake_number
    )

    model_save_dir = os.path.join("model_pickles", f"lake__{lake_number}", index_type)
    os.makedirs(model_save_dir, exist_ok=True)
    
    pickle.dump(model_trained_rf, open(os.path.join(model_save_dir, "rf_model.pkl"), "wb"))
    pickle.dump(model_trained_svm, open(os.path.join(model_save_dir, "svm_model.pkl"), "wb"))
    pickle.dump(model_trained_knn, open(os.path.join(model_save_dir, "knn_model.pkl"), "wb"))
    pickle.dump(model_trained_mlc, open(os.path.join(model_save_dir, "mlc_model.pkl"), "wb"))

    return acc_scores


def plot_combined_contingency_maps(rf_map, svm_map, knn_map, mlc_map, reference_mask, index_type, lake_number):
    """
    Plots contingency maps for RF, SVM, KNN, and MLC models in one row with 4 columns.
    Also computes and exports accuracy metrics (OA, PA, UA, IOU, F1 Score).
    
    Args:
        rf_map (np.array): Classified raster for Random Forest.
        svm_map (np.array): Classified raster for SVM.
        knn_map (np.array): Classified raster for KNN.
        mlc_map (np.array): Classified raster for MLC.
        reference_mask (np.array): Ground truth mask.
        
    Returns:
        dict: Dictionary containing metrics for all models.
    """
    model_maps = {
        "Random Forest": rf_map,
        "SVM": svm_map,
        "KNN": knn_map,
        "MLC": mlc_map
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_dict = {}

    for ax, (model_name, classified_rasters) in zip(axes, model_maps.items()):
        
        # Ensure both masks are binary (0 or 1)
        predicted_mask = (classified_rasters['Combined'] > 0).astype(int)
        reference_mask_binary = (reference_mask > 0).astype(int)

        # Create an empty RGB image
        contingency_map = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

        # True Positive (TP): Blue
        tp = (predicted_mask == 1) & (reference_mask_binary == 1)
        contingency_map[tp] = [0, 0, 255]
        
        # False Positive (FP): Red
        fp = (predicted_mask == 1) & (reference_mask_binary == 0)
        contingency_map[fp] = [255, 0, 0]
        
        # False Negative (FN): Black
        fn = (predicted_mask == 0) & (reference_mask_binary == 1)
        contingency_map[fn] = [0, 0, 0]
        
        # True Negative (TN): White
        tn = (predicted_mask == 0) & (reference_mask_binary == 0)
        contingency_map[tn] = [255, 255, 255]
        
        # Plot the contingency map
        ax.imshow(contingency_map)
        ax.set_title(f"{model_name}", fontsize=15)
        ax.axis("off")

        # Compute metrics
        tn_count = np.sum(tn)
        fp_count = np.sum(fp)
        fn_count = np.sum(fn)
        tp_count = np.sum(tp)
        
        # Metrics Calculation
        oa = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
        pa = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0  # Producer's Accuracy
        ua = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0  # User's Accuracy
        iou = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0
        f1 = (2 * tp_count) / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0

        # Store metrics
        metrics_dict[model_name] = {
            "OA": oa,
            "PA": pa,
            "UA": ua,
            "IOU": iou,
            "F1": f1
        }
        
        # # Print metrics
        # print(f"\nMetrics for {model_name}:")
        # print(f"Overall Accuracy (OA): {oa:.4f}")
        # print(f"Producer's Accuracy (PA): {pa:.4f}")
        # print(f"User's Accuracy (UA): {ua:.4f}")
        # print(f"Intersection over Union (IOU): {iou:.4f}")
        # print(f"F1 Score: {f1:.4f}")

    plt.tight_layout()

    # Save the figure to the current folder
    figure_name = f"contingency2__{index_type}.png"
    plt.savefig(figure_name)
    #plt.close(fig)
        
    print(f"Figure saved to: {figure_name}")
    plt.show() 

    # Save metrics to a JSON file
    
    file_name = f"metrics_{index_type}_lake_{lake_number}.json"
    with open(file_name, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to: {file_name}")

    
    return metrics_dict


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

def compute_pfi_and_get_top3(model, combined_features, sampled_labels, feature_names, model_name, index_type, n_repeats=20, n_runs=5):
    """
    Computes robust PFI by averaging importance scores over multiple runs and returns the top 3 important indices.

    Args:
        model: Trained model.
        combined_features (np.array): Combined feature array for prediction.
        sampled_labels (np.array): Ground truth labels.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.
        index_type (str): Type of index ("index_1" or "index_2").
        n_repeats (int): Number of times to shuffle a feature (higher = more robust).
        n_runs (int): Number of PFI runs to average over.

    Returns:
        list: Top 3 feature names based on PFI.
    """
    # Store all importance results across multiple runs
    all_importances = np.zeros((len(feature_names), n_runs))

    for run in range(n_runs):
        result = permutation_importance(
            model, combined_features, sampled_labels, 
            n_repeats=n_repeats, random_state=42 + run, n_jobs=-1
        )
        all_importances[:, run] = result.importances_mean

    # Average the importances over all runs
    averaged_importances = np.mean(all_importances, axis=1)
    sorted_idx = np.argsort(averaged_importances)[::-1]

    # Plot PFI
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], averaged_importances[sorted_idx])
    plt.xlabel("Mean Permutation Importance (Averaged)")
    plt.title(f"{model_name} - {index_type} PFI (Robust)")
    
    # Save the figure to the current folder
    figure_name = f"contingency_3.png"
    plt.savefig(figure_name)
    plt.show()
    
    print(f"Figure saved to: {figure_name}")
    
    # Return top 3 indices
    top3_indices = [feature_names[i] for i in sorted_idx[:3]]
    return top3_indices


def generate_and_compare_images(loaded_bands, sampled_coords, sampled_labels, reference_mask,lake_no):
    """
    Generates combined images for index_1, index_2, and Top 3 indices.

    Args:
        loaded_bands (dict): Loaded raster layers.
        sampled_coords (np.array): Sampled coordinates.
        sampled_labels (np.array): Ground truth labels.
        reference_mask (np.array): Reference mask for comparison.
    """
    # Choose index_1
    loaded_bands_index_1, features_index_1 = choose_index_and_extract_features(loaded_bands, index_names_1, sampled_coords)
    combined_features_index_1 = np.column_stack([features_index_1[idx] for idx in features_index_1.keys()])
    feature_names_1 = list(features_index_1.keys())
    
    # Train RF Model on index_1 for PFI calculation
    model_rf_index_1 = train_rf_models({"Combined": combined_features_index_1}, sampled_labels)['Combined']
    top3_indices = compute_pfi_and_get_top3(model_rf_index_1, combined_features_index_1, sampled_labels, feature_names_1, "RF", "index_1")

    # Extract Top 3 features from loaded_bands for classification
    top3_bands_index = dictfilt(loaded_bands, top3_indices)
    top3_features = extract_features_from_bands(top3_bands_index, sampled_coords)
    
    # Combine Top 3 features into a single array for training
    combined_top3_features = np.column_stack([top3_features[idx] for idx in top3_features.keys()])
    
    # Choose index_2
    loaded_bands_index_2, features_index_2 = choose_index_and_extract_features(loaded_bands, index_names_2, sampled_coords)
    
    # Generate Images for index_1, index_2, and Top 3 Indices
    print("Generating results for index_1 (All Derived Indices)...")
    generate_combined_image(loaded_bands_index_1, features_index_1, sampled_labels, reference_mask,'all_index',lake_no)
    
    print("Generating results for index_2 (All Bands)...")
    generate_combined_image(loaded_bands_index_2, features_index_2, sampled_labels, reference_mask,'raw_5_bands',lake_no)
    
    print("Generating results for Top 3 Features from PFI...")
    generate_combined_image(top3_bands_index, top3_features, sampled_labels, reference_mask,'top_3',lake_no)

def plot_saved_metrics(folder_path):
    """
    Plots saved metrics from JSON files in the current folder.

    Args:
        folder_path (str): Path to the folder where metrics are saved.
    """
    files = [f for f in os.listdir(folder_path) if f.startswith('metrics_') and f.endswith('.json')]
    all_metrics = {}

    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        
        # Extracting index type and lake number from file name
        file_info = file.replace('metrics_', '').replace('.json', '')
        index_type, lake_number = file_info.split('_lake_')
        
        if lake_number not in all_metrics:
            all_metrics[lake_number] = {}
        
        all_metrics[lake_number][index_type] = metrics

    # Plotting
    metrics_keys = ['OA', 'PA', 'UA', 'IOU', 'F1']
    models = ['Random Forest', 'SVM', 'KNN', 'MLC']
    
    for lake_number, metrics_by_index in all_metrics.items():
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        for idx, metric_name in enumerate(metrics_keys):
            ax = axes[idx]
            
            for index_type, metrics in metrics_by_index.items():
                values = [metrics[model][metric_name] for model in models]
                ax.plot(models, values, label=index_type, marker='o')
            
            ax.set_title(f"{metric_name} for Lake {lake_number}")
            ax.set_xlabel("Models")
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()

        # Save the figure to the current folder
        figure_name = f"comparison_metrics_lake.png"
        plt.savefig(figure_name)
        plt.close(fig)
        
        #print(f"Figure saved to: {figure_name}")
        
        plt.show()





# Get all variables currently defined in the workspace
all_variables = [var for var in globals().copy() if not var.startswith("__")]

# Remove all variables except functions and imported modules
for var in all_variables:
    if not callable(globals()[var]) and not isinstance(globals()[var], type(os)): 
        del globals()[var]

# Confirm clearing is successful
print("All variables cleared successfully!")

folder = 'lake__2024__18'

main_dir = '/misc/stu3/shagun/Projects/data_fusion/My_Own/'

folder_path = os.path.join(main_dir,folder) 
os.chdir(folder_path)

mask_path = os.path.join(folder_path,'constant.tif')



index_names_1 = ['NDWI', 'MNDWI', 'LSWI', 'SWI', 'AWEInsh', 'AWEIsh', 'WI2015', 'MBWI','ANDWI','MuWI-R','RWI','RNDWI','SMBWI','MuWI-C']
index_names_2 = ['B2','B3','B8','B4','B11','B12']

bands = {
    'B2': 'B2.tif', 'B3': 'B3.tif', 'B4': 'B4.tif', 'B8': 'B8.tif',
    'B11': 'B11.tif', 'B12': 'B12.tif', 'NDWI': 'NDWI.tif', 'MNDWI': 'MNDWI.tif',
    'LSWI': 'LSWI.tif', 'SWI': 'SWI.tif', 'AWEInsh': 'AWEInsh.tif', 'AWEIsh': 'AWEIsh.tif',
    'WI2015': 'WI2015.tif', 'MBWI': 'MBWI.tif', 'ANDWI' : 'ANDWI.tif', 'MuWI-R' : 'MuWI-R.tif', 
    'RWI' : 'RWI.tif', 'RNDWI' : 'RNDWI.tif', 'SMBWI' : 'SMBWI.tif', 'MuWI-C' :'MuWI-C.tif', 
    'constant' :'constant.tif'
}



# Load all bands
loaded_bands = {}
for key, filename in bands.items():
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        loaded_bands[key], _ = load_band(filepath)
    else:
        print(f"Warning: {filename} not found.")

reference_mask = loaded_bands['constant']


sampled_coords, sampled_labels, transform = load_mask_and_sample(mask_path)
# ###### Only Extracting Indices #####
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
loaded_bands_index = dictfilt(loaded_bands, index_names_1)


features_per_index = extract_features_from_bands(loaded_bands_index, sampled_coords)


# # Extract features for the selected index
# loaded_bands_index, features_per_index = choose_index_and_extract_features(loaded_bands, index_choice, sampled_coords)


if not os.path.exists('supervised'):
    os.makedirs('supervised')
os.chdir('supervised')

generate_and_compare_images(loaded_bands, sampled_coords, sampled_labels, reference_mask, folder)
plot_saved_metrics(".")

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import os

def get_histogram_stretch_params(band_arrays, lower_percent=2, upper_percent=98):
    """Compute global min/max for histogram stretching across all images."""
    mins, maxs = [], []
    for arr in band_arrays:
        for i in range(arr.shape[0]):
            a, b = np.percentile(arr[i], (lower_percent, upper_percent))
            mins.append(a)
            maxs.append(b)
    return min(mins), max(maxs)

def stretch_histogram(img, vmin, vmax):
    """Apply fixed min/max histogram stretching to an image array."""
    out = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return out

def mask_to_polygons(mask, transform):
    """Convert binary mask to shapely polygons."""
    mask = (mask > 0).astype(np.uint8)
    polygons = []
    for geom, val in shapes(mask, mask=mask, transform=transform):
        if val == 1:
            polygons.append(shape(geom))
    return polygons

def plot_rgb_with_vector_mask(folder1, folder2, title1='Period 1', title2='Period 2'):
    """
    Plot RGB composites for two folders with identical histogram stretching and overlay the ground truth mask as a red vector outline.
    """
    def load_rgb_and_mask(folder):
        band_files = {'B2': 'B2.tif', 'B3': 'B3.tif', 'B4': 'B4.tif', 'constant': 'constant.tif'}
        bands = {}
        for band, fname in band_files.items():
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {fname} in {folder}")
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                if band == 'constant':
                    mask_transform = src.transform
                bands[band] = arr
        rgb = np.stack([bands['B4'], bands['B3'], bands['B2']])
        return rgb, bands['constant'], mask_transform

    rgb1, mask1, transform1 = load_rgb_and_mask(folder1)
    rgb2, mask2, transform2 = load_rgb_and_mask(folder2)

    # Compute global histogram stretch params
    vmin, vmax = get_histogram_stretch_params([rgb1, rgb2])

    # Stretch both images identically
    rgb1_stretched = np.transpose(stretch_histogram(rgb1, vmin, vmax), (1, 2, 0))
    rgb2_stretched = np.transpose(stretch_histogram(rgb2, vmin, vmax), (1, 2, 0))

    # Convert masks to polygons
    polygons1 = mask_to_polygons(mask1, transform1)
    polygons2 = mask_to_polygons(mask2, transform2)

    def geo_to_pixel_coords(x, y, transform):
        """Convert geospatial coordinates to pixel coordinates."""
        col, row = ~transform * (x, y)
        return col, row

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, rgb_img, polygons, transform, title in zip(
        axes, [rgb1_stretched, rgb2_stretched], [polygons1, polygons2], [transform1, transform2], [title1, title2]
    ):
        ax.imshow(rgb_img)
        for poly in polygons:
            if not poly.is_empty:
                x, y = poly.exterior.xy
                px, py = geo_to_pixel_coords(np.array(x), np.array(y), transform)
                ax.plot(px, py, color='red', linewidth=1.5)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_lake_pair(lake_number):
    """
    Plot RGB with vector mask for a given lake number for both time periods.
    """
    folder1 = f'lake__{lake_number}'
    folder2 = f'lake__2024__{lake_number}'
    title1 = f'Lake {lake_number} - Period 1'
    title2 = f'Lake {lake_number} - Period 2'
    plot_rgb_with_vector_mask(folder1, folder2, title1=title1, title2=title2)

def plot_lake_pair_all():
    """
    Plot RGB with vector mask for all lakes (1 to 43) for both time periods.
    """
    for lake_number in range(1, 44):
        print(f"Plotting Lake {lake_number}...")
        plot_lake_pair(lake_number)
        # Optionally, add plt.pause(1) or save figures instead of showing for batch mode

def plot_temporal_transferability_index(metric='IOU'):
    """
    Scatter plot comparing a metric (default IOU) for each model and lake between time 1 and time 2.
    Point size is based on number of water pixels in period 2. Color is model. Diagonal is y=x.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    model_names = ['Random Forest', 'SVM', 'KNN', 'MLC']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    period1_metrics = {}
    period2_metrics = {}
    water_pixel_counts = {}

    # Gather metrics for all lakes
    for lake_number in range(1, 44):
        # File paths
        f1 = f'metrics_all_index_lake_{lake_number}.json'
        f2 = f'metrics_all_index_lake_2024__{lake_number}.json'
        # Check existence
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        # Load metrics
        with open(f1, 'r') as f:
            m1 = json.load(f)
        with open(f2, 'r') as f:
            m2 = json.load(f)
        period1_metrics[lake_number] = m1
        period2_metrics[lake_number] = m2
        # Count water pixels in period 2 mask
        mask_path = os.path.join(f'lake__2024__{lake_number}', 'constant.tif')
        if os.path.exists(mask_path):
            import rasterio
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                water_pixel_counts[lake_number] = np.sum(mask == 1)
        else:
            water_pixel_counts[lake_number] = 0

    # Prepare data for scatter plot
    plt.figure(figsize=(10, 10))
    for model, color in zip(model_names, colors):
        x, y, sizes = [], [], []
        for lake in period1_metrics:
            if model in period1_metrics[lake] and model in period2_metrics[lake]:
                xval = period1_metrics[lake][model][metric]
                yval = period2_metrics[lake][model][metric]
                x.append(xval)
                y.append(yval)
                sizes.append(water_pixel_counts.get(lake, 10))
        x = np.array(x)
        y = np.array(y)
        sizes = np.array(sizes)
        # Normalize sizes for plotting
        if len(sizes) > 0:
            sizes = 100 * (sizes / (np.max(sizes) + 1e-6)) + 30
        plt.scatter(x, y, s=sizes, alpha=0.7, label=model, color=color, edgecolor='k', linewidth=0.5)

    # Diagonal reference
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='y = x')
    plt.xlabel(f'{metric} (Time 1)')
    plt.ylabel(f'{metric} (Time 2)')
    plt.title(f'Temporal Transferability Index ({metric})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
