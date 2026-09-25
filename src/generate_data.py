
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(
    path: str = "data/longitudinal_diabetes.csv",
    seed: int = 42,
    patients: int = 240,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2018-01-01")

    for pnum in range(patients):
        pid = f"P{pnum + 1:04d}"
        risk = rng.normal(0, 1)
        age = int(rng.integers(35, 76))
        bmi0 = float(np.clip(rng.normal(29 + 2.5 * risk, 4), 19, 45))
        hba1c0 = float(np.clip(
            rng.normal(5.7 + 0.45 * risk, 0.45), 4.4, 8.4
        ))
        cond = (
            "hypertension"
            if rng.random() < 0.38 + 0.12 * (risk > 0)
            else "none"
        )

        for visit in range(4):
            dt = start + pd.Timedelta(
                days=180 * visit + int(rng.integers(-10, 11))
            )
            hb = float(np.clip(
                hba1c0 + visit * (0.10 + 0.11 * risk)
                + rng.normal(0, 0.10),
                4.0, 11.5,
            ))
            gl = float(np.clip(
                74 + 24 * hb + rng.normal(0, 8), 70, 300
            ))
            bmi = float(np.clip(
                bmi0 + visit * rng.normal(0.10, 0.08)
                + rng.normal(0, 0.25),
                17, 50,
            ))
            sys_bp = float(np.clip(
                112 + 0.9 * (age - 35) + 2.2 * bmi
                + (8 if cond == "hypertension" else 0)
                + rng.normal(0, 8),
                90, 210,
            ))
            dia_bp = float(np.clip(
                67 + 0.35 * (age - 35) + 0.55 * bmi
                + (5 if cond == "hypertension" else 0)
                + rng.normal(0, 5),
                50, 130,
            ))
            rows.append({
                "patient_id": pid,
                "visit": visit + 1,
                "date": dt.date().isoformat(),
                "age": age,
                "hba1c": round(hb, 3),
                "glucose": round(gl, 2),
                "bmi": round(bmi, 2),
                "systolic_bp": round(sys_bp, 2),
                "diastolic_bp": round(dia_bp, 2),
                "conditions": cond,
            })

    frame = pd.DataFrame(rows)

    # derive progression target
    first_hb = frame[frame.visit == 1].set_index("patient_id").hba1c
    last_hb = frame[frame.visit == 4].set_index("patient_id").hba1c
    labels = (
        ((last_hb - first_hb >= 0.5) | (last_hb >= 6.5))
        .astype(int)
        .rename("progression")
    )
    frame = frame.join(labels, on="patient_id")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


if __name__ == "__main__":
    data = generate_dataset()
    pos = data.groupby("patient_id").progression.first().sum()
    print(
        f"wrote {len(data)} visits for {data.patient_id.nunique()} patients; "
        f"positive patients={pos}"
    )
