import os
import json
import pandas as pd

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "Students Performance .csv")
JSON_FILE_PATH = os.path.join(BASE_DIR, "student_performance.json")


def convert_csv_to_json(csv_path: str = CSV_FILE_PATH, json_path: str = JSON_FILE_PATH):
    """
    Reads the student performance CSV, handles missing values,
    and exports records to a JSON file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Impute missing Scholarship if needed
    if "Scholarship" in df.columns and df["Scholarship"].isna().any():
        mode_scholarship = df["Scholarship"].mode(dropna=True)[0]
        df["Scholarship"] = df["Scholarship"].fillna(mode_scholarship)

    # Clean known anomalies if any
    if "Attendance" in df.columns:
        df["Attendance"] = df["Attendance"].replace({"3": "Never"})
    if "Notes" in df.columns:
        df["Notes"] = df["Notes"].replace({"6": "Yes"})
    if "Listening_in_Class" in df.columns:
        df["Listening_in_Class"] = df["Listening_in_Class"].replace({"6": "Yes"})

    # Export to JSON
    df.to_json(json_path, orient="records", indent=2)
    print(f"Successfully converted '{os.path.basename(csv_path)}' ({len(df)} records) to '{os.path.basename(json_path)}'")
    return df


def load_data(json_path: str = JSON_FILE_PATH):
    """Loads student records from JSON."""
    if not os.path.exists(json_path):
        convert_csv_to_json(json_path=json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def Id_to_student():
    """Maps student records by Student_ID."""
    data = load_data()
    student_dict = {}
    for idx, student in enumerate(data, start=1):
        sid = student.get("Student_ID") or f"STUDENT{idx}"
        student_dict[sid] = student
    return student_dict


if __name__ == "__main__":
    convert_csv_to_json()
    students_map = Id_to_student()
    print(f"Total students indexed: {len(students_map)}")
    print("First record preview:")
    first_key = next(iter(students_map))
    print(json.dumps({first_key: students_map[first_key]}, indent=2))
