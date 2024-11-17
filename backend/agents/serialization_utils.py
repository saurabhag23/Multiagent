# agents/serialization_utils.py

import json
import pandas as pd
import numpy as np
from datetime import datetime

def serialize_pandas(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (dict, list)):
        return json.loads(json.dumps(obj, default=serialize_pandas))
    else:
        return str(obj)

def serialize_result(result):
    return json.loads(json.dumps(result, default=serialize_pandas))