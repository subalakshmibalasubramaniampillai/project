// Shared labels, number formatting and plain-language names.

export const OUTCOMES = {
  readmit30: {
    short: "30-day readmission",
    question: "Will the patient be readmitted within 30 days of discharge?",
    event: "readmitted within 30 days",
  },
  escalation: {
    short: "Treatment escalation",
    question:
      "Will diabetes treatment be stepped up at the patient's next hospital stay?",
    event: "treatment stepped up at the next stay",
  },
};

export const DESIGNS = {
  grouped: {
    short: "New patients",
    long: "Tested on patients the models never saw (5 repeats)",
  },
  temporal: {
    short: "Newest admissions",
    long: "Trained on older admissions, tested on the most recent 20%",
  },
};

export const MODELS = {
  logistic_regression: { name: "Logistic regression", family: "standard" },
  random_forest: { name: "Random forest", family: "standard" },
  xgboost: { name: "XGBoost", family: "standard" },
  lightgbm: { name: "LightGBM", family: "standard" },
  mlp: { name: "Neural network (MLP)", family: "standard" },
  gru: { name: "GRU", family: "sequence" },
  retain: { name: "RETAIN", family: "sequence" },
  transformer: { name: "Transformer", family: "sequence" },
  tkgn: { name: "TKGN", family: "ours" },
  tkgn_b: { name: "TKGN-B", family: "ours" },
};

export const MODEL_ORDER = Object.keys(MODELS);

export const FAMILIES = {
  standard: "Standard machine-learning models",
  sequence: "Published models that read the visit history",
  ours: "Models developed in this project",
};

// chart roles: our final model, the strongest standard model, everything else
export const ROLE_COLOR = {
  ours: "var(--c-ours)",
  reference: "var(--c-ref)",
  other: "var(--c-other)",
};

export function roleOf(model) {
  if (model === "tkgn_b") return "ours";
  if (model === "lightgbm") return "reference";
  return "other";
}

export function num(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toFixed(digits);
}

export function signed(value, digits = 3) {
  if (value === null || value === undefined) return "–";
  const v = Number(value);
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;
}

export function pct(value, digits = 1) {
  if (value === null || value === undefined) return "–";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function count(value) {
  return Number(value).toLocaleString("en-US");
}

export function pText(p) {
  if (p === null || p === undefined) return "–";
  return p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`;
}

// ── ICD-9-CM ───────────────────────────────────────────────

const CHAPTERS = [
  [1, 139, "Infectious disease"],
  [140, 239, "Neoplasm"],
  [240, 279, "Endocrine or metabolic"],
  [280, 289, "Blood disorder"],
  [290, 319, "Mental disorder"],
  [320, 389, "Nervous system"],
  [390, 459, "Circulatory system"],
  [460, 519, "Respiratory system"],
  [520, 579, "Digestive system"],
  [580, 629, "Genitourinary system"],
  [630, 679, "Pregnancy"],
  [680, 709, "Skin"],
  [710, 739, "Musculoskeletal"],
  [740, 759, "Congenital"],
  [760, 779, "Perinatal"],
  [780, 799, "Symptoms and signs"],
  [800, 999, "Injury or complication"],
];

// standard ICD-9-CM category titles for the most frequent codes in the data
const CATEGORY = {
  "038": "Septicaemia",
  "250": "Diabetes mellitus",
  "272": "Lipid disorder",
  "276": "Fluid or electrolyte disorder",
  "285": "Other anaemia",
  "401": "Essential hypertension",
  "402": "Hypertensive heart disease",
  "403": "Hypertensive kidney disease",
  "410": "Acute myocardial infarction",
  "411": "Acute ischaemic heart disease",
  "413": "Angina pectoris",
  "414": "Chronic ischaemic heart disease",
  "424": "Heart valve disease",
  "427": "Cardiac dysrhythmia",
  "428": "Heart failure",
  "433": "Precerebral artery occlusion",
  "434": "Cerebral artery occlusion",
  "435": "Transient cerebral ischaemia",
  "440": "Atherosclerosis",
  "486": "Pneumonia",
  "491": "Chronic bronchitis",
  "493": "Asthma",
  "496": "Chronic airway obstruction",
  "518": "Other lung disease",
  "562": "Diverticula of intestine",
  "574": "Gallstones",
  "577": "Pancreatic disease",
  "584": "Acute kidney failure",
  "585": "Chronic kidney disease",
  "599": "Urinary tract disorder",
  "682": "Cellulitis or abscess",
  "707": "Chronic skin ulcer",
  "715": "Osteoarthritis",
  "722": "Intervertebral disc disorder",
  "780": "General symptoms",
  "786": "Respiratory or chest symptoms",
  "996": "Complication of device or implant",
  "998": "Complication of procedure",
  V57: "Rehabilitation care",
  V45: "Post-procedural status",
};

export function diagnosisName(code) {
  if (!code) return "";
  const text = String(code).trim();
  const head = text.split(".")[0];
  if (CATEGORY[head]) return CATEGORY[head];
  if (text[0] === "V") return "Health-status factor";
  if (text[0] === "E") return "External cause";
  const n = parseInt(head, 10);
  const chapter = CHAPTERS.find(([lo, hi]) => n >= lo && n <= hi);
  return chapter ? chapter[2] : "";
}

export const COMMON_DIAGNOSES = Object.entries(CATEGORY).map(([code, name]) => ({
  code,
  name,
}));

// ── model features in plain words ──────────────────────────

const FEATURE_NAMES = {
  number_inpatient: "Inpatient stays in the previous year",
  number_emergency: "Emergency visits in the previous year",
  number_outpatient: "Outpatient visits in the previous year",
  number_diagnoses: "Number of diagnoses recorded",
  time_in_hospital: "Length of stay (days)",
  num_medications: "Number of medications given",
  num_lab_procedures: "Number of laboratory tests",
  num_procedures: "Number of procedures",
  hist_n_prior: "Number of earlier stays",
  hist_prior_readmit30: "Earlier stays followed by readmission within 30 days",
  hist_prior_readmit_any: "Earlier stays followed by any readmission",
  hist_prior_insulin: "Earlier stays with insulin",
  hist_mean_time_in_hospital: "Average length of earlier stays",
  hist_mean_num_medications: "Average medications at earlier stays",
  hist_mean_number_diagnoses: "Average diagnoses at earlier stays",
  hist_mean_num_lab_procedures: "Average laboratory tests at earlier stays",
  hist_mean_n_active_drugs: "Average number of diabetes drugs at earlier stays",
  delta_n_active_drugs: "Change in number of diabetes drugs since last stay",
  delta_num_medications: "Change in medications since last stay",
  delta_time_in_hospital: "Change in length of stay since last stay",
  delta_number_diagnoses: "Change in diagnoses since last stay",
  delta_num_lab_procedures: "Change in laboratory tests since last stay",
  hist_last_outcome_gt30: "Last stay was followed by readmission after 30 days",
  hist_last_outcome_lt30: "Last stay was followed by readmission within 30 days",
  hist_last_outcome_no: "Last stay was not followed by readmission",
};

const DISCHARGE = {
  1: "Discharged home",
  2: "Transferred to another hospital",
  3: "Discharged to a skilled nursing facility",
  5: "Transferred to other inpatient care",
  6: "Discharged home with home health care",
  7: "Left against medical advice",
  22: "Discharged to a rehabilitation facility",
  23: "Discharged to a long-term care hospital",
};

const DRUG_STATE = { Steady: "steady dose", Up: "dose increased", Down: "dose reduced" };

export function featureName(raw) {
  if (FEATURE_NAMES[raw]) return FEATURE_NAMES[raw];
  const [key, value] = raw.split("=");
  if (value === undefined) return raw.replaceAll("_", " ");
  if (key === "discharge_disposition_id")
    return DISCHARGE[value] || `Discharge type ${value}`;
  if (key === "admission_type_id") return `Admission type ${value}`;
  if (key === "admission_source_id") return `Admission source ${value}`;
  if (key === "payer_code") return value === "<MISSING>" ? "Payer not recorded" : `Payer ${value}`;
  if (key === "medical_specialty")
    return value === "<MISSING>" ? "Admitting specialty not recorded" : `Specialty: ${value}`;
  if (key === "age") return `Age ${value.replace("[", "").replace(")", "")} years`;
  if (key === "diabetesMed") return value === "No" ? "No diabetes medication" : "Diabetes medication prescribed";
  if (key === "change") return value === "Ch" ? "Diabetes medication changed" : "No medication change";
  if (key === "A1Cresult") return value === "None" ? "HbA1c not measured" : `HbA1c ${value}`;
  if (key === "dx") return `Diagnosis ${value} (${diagnosisName(value).toLowerCase()})`;
  if (key === "chapter" || key === "primary_chapter") return `Diagnosis group ${value}`;
  if (DRUG_STATE[value]) return `${key[0].toUpperCase()}${key.slice(1)}, ${DRUG_STATE[value]}`;
  return `${key.replaceAll("_", " ")}: ${value}`;
}

export const INPUT_GROUP_NAMES = {
  "Utilisation counts": "Hospital use (stays, visits, tests)",
  "Prior encounters (sequence)": "The patient's earlier stays",
  "Admission / discharge context": "Admission and discharge details",
  "Diagnoses (ICD-9 codes)": "Diagnoses",
  "Anti-diabetic medications": "Diabetes medicines",
  "Outcomes of prior encounters": "Outcomes of earlier stays",
  Demographics: "Age, sex and race",
  "Medication change flags": "Medication change flags",
  "Glycaemic tests (HbA1c, glucose)": "HbA1c and glucose tests",
};
