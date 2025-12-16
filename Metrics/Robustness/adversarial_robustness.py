from art.estimators.classification import SklearnClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def adversarial_evaluation(X, y, estimator, clip_values=None, attack_class=HopSkipJump, max_iter=10):
    """
    Evaluates one or multiple models (if a list) against an adversarial attack.
    Returns the following metrics:
    - After-attack Accuracy
    - Attack Success Rate (ASR)
    - AUC-ROC (if applicable)
    """
    is_list = isinstance(estimator, list)
    aggregated = []

    # Case 1: Multiple estimators
    if is_list:
        for i, model_dict in enumerate(estimator):
            print(f"\n🔍 Evaluating model {i}: {type(model_dict['estimator'])}")

            try:
                # Use specific X and y for this model
                X_model = model_dict['X']
                y_model = model_dict['y']

                clip_min = float(np.min(X_model))
                clip_max = float(np.max(X_model))
                if clip_min >= clip_max:
                    print(f"❌ Invalid clip_values: min={clip_min}, max={clip_max}")
                    continue

                # Wrap model with ART
                model = model_dict['estimator']
                if model.__class__.__module__.startswith("sklearn"):
                    classifier = SklearnClassifier(model=model, clip_values=(clip_min, clip_max))
                else:
                    raise TypeError(f"Unsupported model type: {type(model)}")

                # Clean prediction
                y_pred_clean = np.argmax(classifier.predict(X_model), axis=1)
                acc_clean = np.mean(y_pred_clean == y_model)

                # Adversarial attack
                attack = attack_class(classifier=classifier, max_iter=max_iter)
                X_adv = attack.generate(x=X_model)

                # Adversarial prediction
                y_pred_adv = np.argmax(classifier.predict(X_adv), axis=1)
                acc_adv = np.mean(y_pred_adv == y_model)
                asr = np.mean(y_pred_clean != y_pred_adv)

                # AUC
                try:
                    y_score_adv = classifier.predict(X_adv)[:, 1]
                    auc = roc_auc_score(y_model, y_score_adv)
                except:
                    auc = None

                # Show metrics
                print(f"✅ After-attack Accuracy : {acc_adv:.3f}")
                print(f"🔥 Attack Success Rate   : {asr:.3f}")
                print(f"🎯 AUC-ROC adversarial   : {auc:.3f}" if auc is not None else "ℹ️ AUC-ROC not available")

                # Store for aggregation
                aggregated.append({
                    'after_attack_accuracy': acc_adv,
                    'attack_success_rate': asr,
                    'auc_roc_adversarial': auc
                })

            except Exception as e:
                print(f"❌ Model {i} incompatible with ART: {e}")

        # Combined summary
        if aggregated:
            print("\n📊 Combined metrics:")
            mean_acc = np.mean([m['after_attack_accuracy'] for m in aggregated])
            mean_asr = np.mean([m['attack_success_rate'] for m in aggregated])
            aucs = [m['auc_roc_adversarial'] for m in aggregated if m['auc_roc_adversarial'] is not None]
            mean_auc = np.mean(aucs) if aucs else None

            print(f"✅ After-attack Accuracy : {mean_acc:.3f}")
            print(f"🔥 Attack Success Rate   : {mean_asr:.3f}")
            print(f"🎯 AUC-ROC adversarial   : {mean_auc:.3f}" if mean_auc is not None else "ℹ️ AUC-ROC not available")

        return aggregated

    # Case 2: Single model
    else:
        if X is None or y is None:
            raise ValueError("You must provide X and y for a single model.")

        clip_min = float(np.min(X))
        clip_max = float(np.max(X))
        if clip_min >= clip_max:
            raise ValueError("Invalid clip_values: min >= max")
        clip_values = (clip_min, clip_max)

        if estimator.__class__.__module__.startswith("sklearn"):
            classifier = SklearnClassifier(model=estimator, clip_values=clip_values)
        else:
            raise TypeError(f"Unsupported model type: {type(estimator)}")

        y_pred_clean = np.argmax(classifier.predict(X), axis=1)
        acc_clean = np.mean(y_pred_clean == y)

        attack = attack_class(classifier=classifier, max_iter=max_iter)
        X_adv = attack.generate(x=X)
        y_pred_adv = np.argmax(classifier.predict(X_adv), axis=1)
        acc_adv = np.mean(y_pred_adv == y)
        asr = np.mean(y_pred_clean != y_pred_adv)

        try:
            y_score_adv = classifier.predict(X_adv)[:, 1]
            auc = roc_auc_score(y, y_score_adv)
        except:
            auc = None

        return {
            'after_attack_accuracy': acc_adv,
            'attack_success_rate': asr,
            'auc_roc_adversarial': auc
        }
