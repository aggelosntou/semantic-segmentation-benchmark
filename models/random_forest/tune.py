import os
import sys
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.random_forest.dataset import load_dataset, prepare_training_data, extract_features
from models.random_forest.model import save_model


# -------------------------------------------------------------
# Utility: sample a percentage of pixels to speed up trials
# -------------------------------------------------------------
def sample_pixels(X, y, fraction=0.10, seed=42):
    n = len(X)
    k = int(n * fraction)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, k, replace=False)
    return X[idx], y[idx]


# -------------------------------------------------------------
# Compute macro-F1 for validation
# -------------------------------------------------------------
def macro_f1(y_true, y_pred, num_classes):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(num_classes)),
        zero_division=0
    )
    return f1.mean()


# -------------------------------------------------------------
# Optuna objective function
# -------------------------------------------------------------
def objective(trial):
    # Load cached dataset (global variables set in main())
    X_train_sub, y_train_sub, X_val, y_val = objective.X_train_sub, objective.y_train_sub, objective.X_val, objective.y_val
    num_classes = objective.num_classes

    # Search space
    n_estimators = trial.suggest_int("n_estimators", 100, 600)
    max_depth = trial.suggest_int("max_depth", 10, 80)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

    # Build model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )

    # Fit on subsampled training set
    clf.fit(X_train_sub, y_train_sub)

    # Evaluate on validation set
    y_pred = clf.predict(X_val)
    score = macro_f1(y_val, y_pred, num_classes)

    return score


# -------------------------------------------------------------
# Main: load data, split, subsample, optimize RF
# -------------------------------------------------------------
def main():
    print("\n[Optuna] Loading full training dataset...")
    data_root = os.path.join(PROJECT_ROOT, "dataset", "train")
    img_dir = os.path.join(data_root, "image")
    mask_dir = os.path.join(data_root, "mask")

    images, masks = load_dataset(img_dir, mask_dir)
    X, y = prepare_training_data(images, masks)
    num_classes = len(np.unique(y))

    print(f"[Optuna] Dataset loaded: X={X.shape}, y={y.shape}, classes={num_classes}")

    # Train/validation split (fixed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"[Optuna] Train: {X_train.shape}, Val: {X_val.shape}")

    # Subsample 10% of training pixels for each trial
    X_train_sub, y_train_sub = sample_pixels(X_train, y_train, fraction=0.10)

    print(f"[Optuna] Subsampled training set: {X_train_sub.shape}")

    # Attach data to objective for global access
    objective.X_train_sub = X_train_sub
    objective.y_train_sub = y_train_sub
    objective.X_val = X_val
    objective.y_val = y_val
    objective.num_classes = num_classes

    # Run optimization
    print("\n[Optuna] Starting optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print("\n[Optuna] Study Completed.")
    print(f"Best Macro-F1 = {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f" - {k}: {v}")

    # Train final model on *full* training set with best params
    print("\n[Optuna] Training final model on FULL training set...")
    best_clf = RandomForestClassifier(
        **study.best_params,
        n_jobs=-1,
        random_state=42,
    )
    best_clf.fit(X_train, y_train)

    out_dir = os.path.join(PROJECT_ROOT, "scripts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rf_model_tuned.pkl")
    save_model(best_clf, out_path)

    print(f"[Optuna] Saved tuned model → {out_path}")


if __name__ == "__main__":
    main()
