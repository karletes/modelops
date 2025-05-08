# Overview

Simple Logistic Regression model for a medical dataset to identify propensity to a heart stroke.


# Data Model

```sql
CREATE MULTISET TABLE demo.stroke_data
(
    id INTEGER,
    gender VARCHAR(20) CHARACTER SET LATIN NOT CASESPECIFIC,
    age DECIMAL(6,2),
    hypertension SMALLINT,
    heart_disease SMALLINT,
    ever_married VARCHAR(20) CHARACTER SET LATIN NOT CASESPECIFIC,
    work_type VARCHAR(20) CHARACTER SET LATIN NOT CASESPECIFIC,
    Residence_type VARCHAR(20) CHARACTER SET LATIN NOT CASESPECIFIC,
    avg_glucose_level DECIMAL(8,2),
    bmi DECIMAL(8,2),
    smoking_status VARCHAR(20) CHARACTER SET LATIN NOT CASESPECIFIC,
    stroke SMALLINT
)
PRIMARY INDEX ( id );
```

```sql
CREATE MULTISET TABLE demo.stroke_preds
(
    job_id VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
    id BIGINT,
    stroke INTEGER,
    json_report CLOB(1048544000) CHARACTER SET UNICODE
)
UNIQUE PRIMARY INDEX (job_id,id);
```


# Scoring 

This demo mode supports two types of scoring:

 - Batch
 - RESTful

In-Vantage scoring is not supported, and would need a PMML file produceds during training.

Batch Scoring is supported via the `score` method in [scoring.py](model_modules/scoring.py).
Scores are appended to `demo.stroke_preds` table.

RESTful scoring is supported via the `ModelScorer` class which implements a predict method which is called by the RESTful Serving Engine. An example request is:

    curl -X POST http://<service-name>/predict \
        -H "Content-Type: application/json" \
        -d '{
          "data": {
            "ndarray": [[
              6,
              148,
              72,
              35,
              0,
              33.6,
              0.627,
              50
            ]],
            "names":[
              "NumTimesPrg",
              "PlGlcConc",
              "BloodP",
              "SkinThick",
              "TwoHourSerIns",
              "BMI",
              "DiPedFunc",
              "Age"
            ]
          }
        }' 
