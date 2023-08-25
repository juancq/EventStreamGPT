#!/usr/bin/env python
# coding: utf-8

# # Synthetic Data Generation
# 
# This notebook generates some simple synthetic data for us to use to demonstrate the ESGPT pipeline. We'll generate a few files:
#   1. `subjects.parquet`, which contains static data about each subject.
#   2. `admit.csv`, which contains records of admissions.
#   
# This is all synthetic data designed solely for demonstrating this pipeline. It is *not* real data, derived from real data, or designed to mimic real data in any way other than plausible file structure.

# In[1]:


import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from tqdm import tqdm

random.seed(1)
np.random.seed(1)


# In[2]:


# Parameters:
N_subjects = int(9e6)
OUT_DIR = Path("./raw")


# ## Subjects Data
# Subjects will have the following static data elements, and will be organized by a fake identifier column called "MRN":
#   * Date of birth
#   * sex

# In[3]:


random.seed(1)
np.random.seed(1)

BASE_BIRTH_DATE = datetime(1950, 1, 1)
SEX = ["male", "female"]

def yrs_to_dob(yrs: np.ndarray) -> list[str]:
    return [(BASE_BIRTH_DATE + timedelta(days=365 * x)).strftime("%m/%d/%Y") for x in yrs]


size = (N_subjects,)
subject_data = pl.DataFrame(
    {
        "MRN": np.arange(900, 900+size[0]).astype(str),
        "dob": yrs_to_dob(np.random.uniform(low=-10, high=10, size=size)),
        "sex": list(np.random.choice(SEX, size=size, replace=True)),
    }
).sample(fraction=1, with_replacement=False, shuffle=True, seed=1)

assert len(subject_data["MRN"].unique()) == N_subjects

subject_data.write_parquet(OUT_DIR / "subjects.parquet", use_pyarrow=True)
print(subject_data.head(3))

# ## Admission admin Data
# This file will contain records of admission start and end dates, and other administrative data

# In[4]:


random.seed(1)
np.random.seed(1)

admit_data = {
    "MRN": [],
    "admit_date": [],
    "disch_date": [],
    "hospital_type": [],
    "acute_flag": [],
    "episode_care_type": [],
    "facility_type": [],
    "procedure1": [],
    "procedure2": [],
    "procedure3": [],
    "er_status": [],
    "mode_separation": [],
    "insurance": [],
}

BASE_ADMIT_DATE = datetime(2002, 1, 1)

hrs = 60
days = 24 * hrs
months = 30 * days

n_admissions_L = np.random.randint(low=1, high=10, size=size)

letters = "abcdefghijklmnopqrstuvwxyz"
admissions_by_subject = {}
get_procedure_code = lambda: ''.join(random.choices(letters[:-3], k=3))

for MRN, n_admissions in tqdm(zip(subject_data["MRN"], n_admissions_L)):
    admit_gaps = np.random.uniform(low=1 * days, high=6 * months, size=(n_admissions,))
    admit_lens = np.random.uniform(low=12 * hrs, high=14 * days, size=(n_admissions,))

    hospital_type_L = np.random.randint(low=1, high=3, size=n_admissions).astype(str)
    acute_flag_L = np.random.choice(['Y', 'N', 'UNK'], size=n_admissions, replace=True, p=[0.4, 0.4, 0.2])
    episode_care_type_L = np.random.choice(list(map(str, range(10))) + ['M'], size=n_admissions, replace=True)

    facility_type_L = [random.choice(letters) + str(np.random.randint(1, 3)) for _ in range(n_admissions)]
    # 12167 possible procedure codes
    procedure1_L = [get_procedure_code() for _ in range(n_admissions)]

    procedure2_L = np.full(n_admissions, '')
    _size = int(n_admissions*0.5)
    procedure2_L[np.random.choice(list(range(n_admissions)), size=_size)] = [get_procedure_code() for _ in range(_size)]

    procedure3_L = np.full(n_admissions, '')
    _size = int(n_admissions*0.2)
    procedure3_L[np.random.choice(list(range(n_admissions)), size=_size)] = [get_procedure_code() for _ in range(_size)]

    er_status_L = np.random.choice(list(map(str, range(1,6))), size=n_admissions, replace=True)
    mode_separation_L = np.random.choice(list(map(str, range(12))), size=n_admissions, replace=True)

    insurance_L = np.random.choice(list(map(str, range(10))) + ["other"], size=n_admissions, replace=True)

    running_end = BASE_ADMIT_DATE
    admissions_by_subject[MRN] = []

    for j, (gap, L) in enumerate(zip(admit_gaps, admit_lens)):
        running_start = running_end + timedelta(minutes=gap)
        running_end = running_start + timedelta(minutes=L)

        admissions_by_subject[MRN].append((running_start, running_end))

        vitals_time = running_start
        admit_data["MRN"].append(MRN)
        admit_data["admit_date"].append(running_start.strftime("%m/%d/%Y, %H:%M:%S"))
        admit_data["disch_date"].append(running_end.strftime("%m/%d/%Y, %H:%M:%S"))
        admit_data["hospital_type"].append(hospital_type_L[j])
        admit_data["acute_flag"].append(acute_flag_L[j])
        admit_data["episode_care_type"].append(episode_care_type_L[j])
        admit_data["facility_type"].append(facility_type_L[j])
        admit_data["procedure1"].append(procedure1_L[j])
        admit_data["procedure2"].append(procedure2_L[j])
        admit_data["procedure3"].append(procedure3_L[j])
        admit_data["er_status"].append(er_status_L[j])
        admit_data["mode_separation"].append(mode_separation_L[j])
        admit_data["insurance"].append(insurance_L[j])


admit_data = pl.LazyFrame(admit_data)
admit_data.sink_parquet(OUT_DIR / "admit.parquet")
print(admit_data.head(3).collect())
