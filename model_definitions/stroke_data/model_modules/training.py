from aoa import (
    ModelContext,
    tmo_create_context,
    record_training_stats,
    save_plot
)
from teradataml import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import joblib


class CustomPrep:
    def fit(self, *args, **kwargs): return self
    def transform(self, ds, *args, **kwargs):
        
        # rename columns
        df = ds.rename(columns={
            'smoking_status': 'smoking',
            'Residence_type': 'residence',
            'avg_glucose_level': 'glucose'
            })
        
        # incorrect values
        df = df[df["gender"]!="Other"]

        # missing values
        df["age_group"] = pd.cut(df["age"], bins=list(range(0, 100, 10)), right=False, include_lowest=False)
        df["bmi"] = df.groupby(["gender", "age_group"],observed=False)["bmi"].transform(lambda x: x.fillna(x.mean().round(2)))
        df = df.drop(columns="age_group")

        # binary categories to boolean (1/0)
        df['gender'] = (df['gender'] == 'Male').astype(int)
        df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
        df['residence'] = (df['residence'] == 'Urban').astype(int)
        df.rename(columns={'gender': 'gender_Male', 'residence': 'residence_Urban'}, inplace=True)

        # multi-value categories as dummy columns
        df = pd.get_dummies(df, columns=['work_type', 'smoking'], dtype=int)
        df.drop(columns=['work_type_children','work_type_Never_worked','smoking_Unknown'], inplace=True)

        # standarize continuous columns
        df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
        df['bmi'] = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()
        df['glucose'] = (df['glucose'] - df['glucose'].mean()) / df['glucose'].std()
        
        return df


def train(context: ModelContext, **kwargs):
    tmo_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # 1.
    print("Importing data...")

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    # 2.
    print("Training model...")

    model = Pipeline([
        ('dataprep', CustomPrep()),
        ('regression', LogisticRegression(
            class_weight={0:1,1:context.hyperparams['target_weight']},
            random_state=context.hyperparams['random_state']
        ))
    ])
    model.fit(X_train, y_train)

    print("Finished training")

    # 3.
    joblib.dump(model, f"{context.artifact_output_path}/model.joblib")

    print("Saved trained model")

    # 4.
    feature_importance = pd.DataFrame({
        'feature': model.named_steps['regression'].feature_names_in_,
        'importance': abs(model.named_steps['regression'].coef_[0])

    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance['feature'], feature_importance['importance']
    )
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Logistic Regression')
    save_plot('feature_importance.png', context=context)

    record_training_stats(train_df,
                          features=feature_importance['feature'],
                          targets=[target_name],
                          categorical=[target_name],
                          feature_importance=feature_importance['importance'],
                          context=context)

    print("Saved artifacts")

