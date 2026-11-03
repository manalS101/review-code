# Getting the data

No patient data is included in this repository, and none may be added to it. The
framework reads the **MIMIC-IV-Ext Clinical Decision Making (MIMIC-CDM)** dataset,
which is credentialed-access data hosted on PhysioNet. Redistribution is
prohibited by the data use agreement, so every user must obtain their own copy.

## 1. Get credentialed on PhysioNet

1. Create an account at https://physionet.org/register/.
2. Complete the CITI "Data or Specimens Only Research" training course and
   submit the completion report to PhysioNet. This is the step that takes time;
   allow a few days for review.
3. Sign the data use agreement for MIMIC-IV at
   https://physionet.org/content/mimiciv/2.2/.

## 2. Download the dataset

There are two routes. The first is much less work.

### Route A: download MIMIC-CDM directly (recommended)

Download the pre-built dataset from
https://physionet.org/content/mimic-iv-ext-cdm/ and convert it with the
authors' script:

```bash
git clone https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset
cd MIMIC-Clinical-Decision-Making-Dataset
pip install --no-deps -r requirements.txt
# set base_new at the top of ConvertPhysionet.py to your download folder
python ConvertPhysionet.py
```

### Route B: build it from raw MIMIC-IV

Download the hospital module from
https://physionet.org/content/mimiciv/2.2/ and the notes from
https://www.physionet.org/content/mimic-iv-note/2.2/, then set `base_mimic` and
`base_new` in `CreateDataset.py` in the same repository and run it. This takes
considerably longer and produces the same result.

## 3. Place the files

Copy the converted files into `data/` at the root of this repository:

```
data/
├── history_of_present_illness.csv
├── physical_examination.csv
├── normalized_laboratory_tests.csv
├── microbiology.csv
├── radiology_reports.csv
├── discharge_diagnosis.csv
├── discharge_procedures.csv
├── icd_diagnosis.csv
├── icd_procedures.csv
├── pathology_ids.json
└── lab_test_mapping.csv
```

`pathology_ids.json` maps each condition to its list of admission ids and is
what defines the four condition groups. `lab_test_mapping.csv` maps laboratory labels to item
ids and is what lets the laboratory tool resolve a request such as "CBC" into
actual results. Both ship with the dataset.

If you keep the files somewhere else, pass `--data-dir` and
`--lab-mapping-csv`.

The adapter checks for all of these at construction and fails immediately with
the list of what is missing, rather than producing empty records.

## 4. Confirm the dataset loads

```bash
python -c "from src.adapter import Adapter; a = Adapter('data'); print({k: len(v) for k, v in a.all_hadm_ids().items()})"
```

You should see a count for each of the four conditions.

## Handling rules

- `data/` is in `.gitignore` and must stay there. Do not commit any file derived
  from the dataset, including run outputs.
- Run outputs under `runs/` contain record text and admission ids. They are
  derived data and fall under the same agreement. `runs/` is also ignored.
- Do not send record text to a third-party inference API. The framework is
  built against a self-hosted vLLM endpoint for this reason.
- Check your institution's own requirements before running. Approval to use
  MIMIC for one project does not automatically extend to another.
