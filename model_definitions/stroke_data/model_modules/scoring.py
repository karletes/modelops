from aoa import (
    ModelContext,
    tmo_create_context,
    record_scoring_stats
)
from teradataml import copy_to_sql, DataFrame
import pandas as pd
import joblib


def score(context: ModelContext, **kwargs):
    tmo_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # 1.
    print("Importing data...")

    # read training dataset from Teradata and convert to pandas
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)

    # 2.
    print("Scoring...")
    predictions_pdf = model.predict(features_pdf[feature_names])

    print("Finished Scoring")

    # 3.
    # store the predictions
    predictions_pdf = pd.DataFrame(predictions_pdf, columns=[target_name])
    predictions_pdf[entity_key] = features_pdf.index.values
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_pdf["job_id"] = context.job_id

    # teradataml doesn't match column names on append.. and so to match / use same table schema as for byom predict
    # example (see README.md), we must add empty json_report column and change column order manually (v17.0.0.4)
    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append",
                primary_index=["job_id", "id"], # Not possible to create UPI here, using next best thing
                set_table=True)

    print("Saved to Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
        WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self):
        self.model = joblib.load("artifacts/input/model.joblib")

    def predict(self, data):
        return self.model.predict(data)
