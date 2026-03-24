from utils import set_seed
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from bucket_utils import resume_checkpoint_if_exists
from util_types import ModelName, PoolingStrategy

import numpy as np

def train_linear_model(
    layer_no: int, 
    model_name: ModelName, 
    pool_type: PoolingStrategy,
) -> Tuple[np.ndarray, float]:
    set_seed(42)

    train_idx, train_x_dict, train_y_arr, _ = resume_checkpoint_if_exists(
        split='train', target_layers=[layer_no], pool_type=pool_type, model_name=model_name
    )
    if train_idx == 0:
        raise FileNotFoundError("No checkpoint found on GCS for the train split. Please run extraction first.")
        
    print(f"Using train data up to index {train_idx}")

    # have to slice here because the array was initialized with all zeros as the size of the dataset
    # so slice off the extra zeros
    train_x = train_x_dict[layer_no][:train_idx]
    train_y = train_y_arr[:train_idx]

    test_idx, test_x_dict, test_y_arr, _ = resume_checkpoint_if_exists(
        split='test', target_layers=[layer_no], pool_type=pool_type, model_name=model_name
    )
    if test_idx == 0:
        raise FileNotFoundError("No checkpoint found on GCS for the test split. Please run extraction first.")
        
    print(f"Using test data up to index {test_idx}")
    # have to slice here because the array was initialized with all zeros as the size of the dataset
    # so slice off the extra zeros
    test_x = test_x_dict[layer_no][:test_idx]
    test_y = test_y_arr[:test_idx]

    print(f"\n--- Diagnostic Check for Layer {layer_no} ---")
    print(f"train_x shape: {train_x.shape}")
    std_across_samples = np.std(train_x, axis=0).mean()
    print(f"Mean feature standard deviation across samples: {std_across_samples:.6f}")
    if len(train_x) > 1:
        is_identical = np.allclose(train_x[0], train_x[1])
        print(f"Are sample 0 and sample 1 exactly identical? {'Yes!!' if is_identical else 'No'}")
    print("---------------------------------------------")

    # normalize data
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    print(f"Train label distribution: Safe (0): {len(train_y) - np.sum(train_y)} | Unsafe (1): {np.sum(train_y)}")
    print(f"Test label distribution:  Safe (0): {len(test_y) - np.sum(test_y)} | Unsafe (1): {np.sum(test_y)}")

    chosen_classifier = None
    
    # train linear model with no regularization and different regularizations
    for c_val in [0.001]:
        print(f"\n--- C={c_val if c_val is not None else 'None (No Regularization)'} ---")
        if c_val is None:
            classifier = LogisticRegression(max_iter=500, penalty=None)
        else:
            classifier = LogisticRegression(max_iter=500, C=c_val)
            
        classifier.fit(train_x, train_y)
        
        train_preds = classifier.predict(train_x)
        test_preds = classifier.predict(test_x)
        
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_y, train_preds).ravel()
        fpr_train = fp_train / (fp_train + tn_train) if (fp_train + tn_train) > 0 else 0.0
        fnr_train = fn_train / (fn_train + tp_train) if (fn_train + tp_train) > 0 else 0.0
        print(f"Train - FPR: {fpr_train:.4f} | FNR: {fnr_train:.4f}")
        
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_y, test_preds).ravel()
        fpr_test = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0.0
        fnr_test = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0.0
        print(f"Test  - FPR: {fpr_test:.4f} | FNR: {fnr_test:.4f}")

        # take last one
        chosen_classifier = classifier

    # scikit-learn returns them as arrays of shape (1, n_features) and (1,) for binary classification
    weight = chosen_classifier.coef_[0]
    bias = float(chosen_classifier.intercept_[0])
    
    return weight, bias
