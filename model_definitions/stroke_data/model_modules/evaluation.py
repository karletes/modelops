from aoa import (
    ModelContext,
    tmo_create_context,
    record_evaluation_stats,
    save_plot
)
from teradataml import DataFrame, copy_to_sql
from sklearn import metrics
import shap  # for model explainability
import pandas as pd
import numpy as np
import joblib
import json


def evaluate(context: ModelContext, **kwargs):
    tmo_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
 
    # 1.
    print("Importing data...")

    # read training dataset from Teradata and convert to pandas
    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)

    # split data into X and y
    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name]

    # 2.
    print("Scoring test data...")

    y_pred = model.predict(X_test)
    y_pred_tdf = pd.DataFrame(y_pred, columns=[entity_key, target_name])

    # 3.
    print("Calculating metrics...")

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_plot('Confusion Matrix', context=context)

    metrics.RocCurveDisplay.from_predictions(y_test, y_pred)
    save_plot('ROC Curve', context=context)

    # 4.
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, plot_size=(12, 8), plot_type='bar')
    save_plot('SHAP Feature Importance', context=context)

    feature_importance = pd.DataFrame(list(zip(feature_names, np.abs(shap_values).mean(0))),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance = feature_importance.set_index("col_name").T.to_dict(orient='records')[0]

    predictions_table = "tmp_evaluation_preds"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    record_evaluation_stats(features_df=test_df,
                            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
                            importance=feature_importance,
                            context=context)

    print("Saved artifacts")
