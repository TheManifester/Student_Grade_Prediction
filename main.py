import os
import json
import joblib
import pandas as pd
from typing import Annotated, Optional, Literal, Dict
from pydantic import BaseModel, Field, model_validator
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="Student Grade Prediction & Management API",
    description="End-to-end Student Performance Classification System with Scikit-Learn and FastAPI.",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Students Performance .csv")
DATA_FILE = os.path.join(BASE_DIR, "student_performance.json")
MODEL_FILE = os.path.join(BASE_DIR, "student_classifier_api.pkl")
BACKUP_MODEL_FILE = os.path.join(BASE_DIR, "student_grade_classifier.pkl")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
HTML_FILE = os.path.join(TEMPLATES_DIR, "index.html")

# Cached model instance
_model = None


def get_model():
    """Loads and caches the trained Scikit-Learn classification pipeline."""
    global _model
    if _model is None:
        target_path = MODEL_FILE if os.path.exists(MODEL_FILE) else BACKUP_MODEL_FILE
        if not os.path.exists(target_path):
            raise HTTPException(
                status_code=500,
                detail=f"Model file '{os.path.basename(MODEL_FILE)}' not found. Please run 'train_model.py' first."
            )
        try:
            _model = joblib.load(target_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load trained model: {str(e)}"
            )
    return _model


def load_data():
    """Loads student records from JSON; auto-converts from CSV if JSON does not exist."""
    if not os.path.exists(DATA_FILE):
        if os.path.exists(CSV_FILE):
            from convert_data import convert_csv_to_json
            convert_csv_to_json(CSV_FILE, DATA_FILE)
        else:
            with open(DATA_FILE, "w") as f:
                json.dump([], f)
            return []
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    return data


def save_data(data):
    """Saves student records to student_performance.json."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def Id_to_student():
    """Returns a dictionary mapping student IDs to student record objects."""
    data = load_data()
    student_dict = {}
    for idx, s in enumerate(data, start=1):
        sid = s.get("Student_ID") or f"STUDENT{idx}"
        student_dict[sid] = s
    return student_dict


# ==========================================
# Pydantic Schemas
# ==========================================

class StudentPredictionInput(BaseModel):
    Student_Age: str = Field(..., description="Age group: '18', '19-22', '23-27'", examples=["19-22"])
    Sex: str = Field(..., description="Gender: 'Male' or 'Female'", examples=["Male"])
    High_School_Type: str = Field(..., description="High school type: 'State', 'Private', or 'Other'", examples=["Other"])
    Scholarship: str = Field(default="50%", description="Scholarship: '25%', '50%', '75%', '100%'", examples=["50%"])
    Additional_Work: str = Field(default="No", description="Additional work: 'Yes' or 'No'", examples=["Yes"])
    Sports_activity: str = Field(default="No", description="Sports activity: 'Yes' or 'No'", examples=["No"])
    Transportation: str = Field(default="Private", description="Transportation: 'Private' or 'Bus'", examples=["Private"])
    Weekly_Study_Hours: float = Field(..., ge=0, description="Weekly study hours: e.g. 0, 2, 8, 12", examples=[2])
    Attendance: str = Field(default="Always", description="Attendance: 'Always', 'Sometimes', 'Never'", examples=["Always"])
    Reading: str = Field(default="Yes", description="Reading habit: 'Yes' or 'No'", examples=["Yes"])
    Notes: str = Field(default="Yes", description="Notes taking: 'Yes' or 'No'", examples=["Yes"])
    Listening_in_Class: str = Field(default="No", description="Listening in class: 'Yes' or 'No'", examples=["No"])
    Project_work: str = Field(default="No", description="Project work: 'Yes' or 'No'", examples=["No"])
    Student_ID: Optional[str] = Field(default=None, description="Optional Student ID", examples=["STUDENT1"])

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data):
        if not isinstance(data, dict):
            return data
        key_mapping = {
            "student_age": "Student_Age",
            "studentage": "Student_Age",
            "age": "Student_Age",
            "sex": "Sex",
            "gender": "Sex",
            "high_school_type": "High_School_Type",
            "highschooltype": "High_School_Type",
            "scholarship": "Scholarship",
            "additional_work": "Additional_Work",
            "additionalwork": "Additional_Work",
            "sports_activity": "Sports_activity",
            "sportsactivity": "Sports_activity",
            "sports": "Sports_activity",
            "transportation": "Transportation",
            "weekly_study_hours": "Weekly_Study_Hours",
            "weeklystudyhours": "Weekly_Study_Hours",
            "study_hours": "Weekly_Study_Hours",
            "studyhours": "Weekly_Study_Hours",
            "attendance": "Attendance",
            "reading": "Reading",
            "notes": "Notes",
            "listening_in_class": "Listening_in_Class",
            "listeninginclass": "Listening_in_Class",
            "project_work": "Project_work",
            "projectwork": "Project_work",
            "student_id": "Student_ID",
            "studentid": "Student_ID"
        }
        normalized = {}
        for k, v in data.items():
            norm_k = key_mapping.get(k.lower(), k)
            if isinstance(v, str):
                v_clean = v.strip()
                if norm_k in [
                    "Sex", "High_School_Type", "Additional_Work",
                    "Sports_activity", "Transportation", "Attendance",
                    "Reading", "Notes", "Listening_in_Class", "Project_work"
                ]:
                    v_clean = v_clean.capitalize()
                if norm_k == "Scholarship" and not v_clean.endswith("%") and v_clean.isdigit():
                    v_clean = f"{v_clean}%"
                normalized[norm_k] = v_clean
            else:
                normalized[norm_k] = v
        return normalized


class PredictionResponse(BaseModel):
    status: str = "success"
    student_id: Optional[str] = None
    predicted_grade: str
    confidence: float
    confidence_percentage: str
    class_probabilities: Dict[str, float]


class Student(BaseModel):
    Student_ID: Annotated[str, Field(..., description="Unique student ID", examples=["STUDENT1"])]
    Student_Age: Annotated[str, Field(..., description="Age group", examples=["19-22"])]
    Sex: Annotated[str, Field(..., description="Sex", examples=["Male"])]
    High_School_Type: Annotated[str, Field(..., description="School type", examples=["Other"])]
    Scholarship: Annotated[str, Field(default="50%", examples=["50%"])]
    Additional_Work: Annotated[str, Field(default="No", examples=["Yes"])]
    Sports_activity: Annotated[str, Field(default="No", examples=["No"])]
    Transportation: Annotated[str, Field(default="Private", examples=["Private"])]
    Weekly_Study_Hours: Annotated[float, Field(..., ge=0, examples=[2])]
    Attendance: Annotated[str, Field(default="Always", examples=["Always"])]
    Reading: Annotated[str, Field(default="Yes", examples=["Yes"])]
    Notes: Annotated[str, Field(default="Yes", examples=["Yes"])]
    Listening_in_Class: Annotated[str, Field(default="No", examples=["No"])]
    Project_work: Annotated[str, Field(default="No", examples=["No"])]
    Grade: Annotated[Optional[str], Field(default=None, examples=["AA"])]

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data):
        return StudentPredictionInput.normalize_fields(data)


class Update_Student(BaseModel):
    Student_ID: Optional[str] = None
    Student_Age: Optional[str] = None
    Sex: Optional[str] = None
    High_School_Type: Optional[str] = None
    Scholarship: Optional[str] = None
    Additional_Work: Optional[str] = None
    Sports_activity: Optional[str] = None
    Transportation: Optional[str] = None
    Weekly_Study_Hours: Optional[float] = None
    Attendance: Optional[str] = None
    Reading: Optional[str] = None
    Notes: Optional[str] = None
    Listening_in_Class: Optional[str] = None
    Project_work: Optional[str] = None
    Grade: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data):
        return StudentPredictionInput.normalize_fields(data)


# ==========================================
# Web Interface Route (GET /)
# ==========================================

@app.get("/", response_class=FileResponse)
def index():
    """Serves the interactive web interface from templates/index.html."""
    if not os.path.exists(HTML_FILE):
        raise HTTPException(
            status_code=404,
            detail="Frontend template 'templates/index.html' not found."
        )
    return FileResponse(HTML_FILE, media_type="text/html")


# ==========================================
# POST /predict Endpoint
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
def predict_grade(student: StudentPredictionInput):
    """
    Predicts student grade from input attributes using the serialized
    Scikit-Learn classification pipeline ('student_classifier_api.pkl').
    Returns the predicted grade and confidence percentage.
    """
    model = get_model()

    input_df = pd.DataFrame([{
        "Student_Age": student.Student_Age,
        "Sex": student.Sex,
        "High_School_Type": student.High_School_Type,
        "Scholarship": student.Scholarship,
        "Additional_Work": student.Additional_Work,
        "Sports_activity": student.Sports_activity,
        "Transportation": student.Transportation,
        "Weekly_Study_Hours": float(student.Weekly_Study_Hours),
        "Attendance": student.Attendance,
        "Reading": student.Reading,
        "Notes": student.Notes,
        "Listening_in_Class": student.Listening_in_Class,
        "Project_work": student.Project_work
    }])

    try:
        prediction = model.predict(input_df)[0]
        confidence_float = 0.0
        probabilities = {}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
            max_prob = float(max(proba))
            confidence_float = round(max_prob * 100, 2)

        return PredictionResponse(
            status="success",
            student_id=student.Student_ID,
            predicted_grade=str(prediction),
            confidence=confidence_float,
            confidence_percentage=f"{confidence_float:.1f}%",
            class_probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ==========================================
# Student CRUD Endpoints (Preserved & Matched)
# ==========================================

@app.get("/students")
def view_students():
    """Returns all student performance records from JSON."""
    return load_data()


@app.get("/students/search")
def search_student(student_id: Optional[str] = Query(default=None, description="Search by Student ID")):
    """Searches for a student by Student_ID."""
    if not student_id:
        raise HTTPException(status_code=400, detail="Query parameter 'student_id' is required")

    data = Id_to_student()
    for key, value in data.items():
        if key.lower() == student_id.lower() or str(value.get("Student_ID", "")).lower() == student_id.lower():
            return value

    raise HTTPException(status_code=404, detail="Student not found in data")


@app.get("/students/{student_id}")
def view_student_by_id(student_id: str):
    """Retrieves a single student by Student_ID."""
    data = Id_to_student()
    if student_id in data:
        return data[student_id]
    for key, val in data.items():
        if key.lower() == student_id.lower() or str(val.get("Student_ID", "")).lower() == student_id.lower():
            return val
    raise HTTPException(status_code=404, detail="Student is not in data")


@app.post("/add_student")
def create_student(student: Student):
    """Adds a new student record; automatically predicts grade if not supplied."""
    data = load_data()
    sid = student.Student_ID
    if any(str(s.get("Student_ID", "")).lower() == sid.lower() for s in data):
        raise HTTPException(status_code=400, detail=f"Student '{sid}' already exists")

    student_dict = student.model_dump(by_alias=True)

    if not student_dict.get("Grade"):
        try:
            model = get_model()
            input_df = pd.DataFrame([{
                "Student_Age": student.Student_Age,
                "Sex": student.Sex,
                "High_School_Type": student.High_School_Type,
                "Scholarship": student.Scholarship,
                "Additional_Work": student.Additional_Work,
                "Sports_activity": student.Sports_activity,
                "Transportation": student.Transportation,
                "Weekly_Study_Hours": float(student.Weekly_Study_Hours),
                "Attendance": student.Attendance,
                "Reading": student.Reading,
                "Notes": student.Notes,
                "Listening_in_Class": student.Listening_in_Class,
                "Project_work": student.Project_work
            }])
            predicted = model.predict(input_df)[0]
            student_dict["Grade"] = str(predicted)
        except Exception:
            pass

    data.append(student_dict)
    save_data(data)
    return JSONResponse(status_code=200, content={"message": "Student is listed", "student": student_dict})


@app.put("/update/{student_id}")
def update_student(student_id: str, update_stud: Update_Student):
    """Updates fields of an existing student."""
    raw_data = load_data()
    target_idx = None
    for idx, s in enumerate(raw_data):
        if str(s.get("Student_ID", "")).lower() == student_id.lower():
            target_idx = idx
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail="Student is not listed")

    existing_student = raw_data[target_idx]
    update_info = update_stud.model_dump(exclude_unset=True, exclude_none=True)
    existing_student.update(update_info)
    raw_data[target_idx] = existing_student
    save_data(raw_data)
    return {"message": "Student updated successfully", "student_id": student_id, "student": existing_student}


@app.delete("/delete_student")
def delete(student_id: str):
    """Deletes a student record by Student_ID."""
    raw_data = load_data()
    target_idx = None
    for idx, s in enumerate(raw_data):
        if str(s.get("Student_ID", "")).lower() == student_id.lower():
            target_idx = idx
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail="Student is not listed")

    deleted = raw_data.pop(target_idx)
    save_data(raw_data)
    return JSONResponse(status_code=200, content={"message": "Student is deleted", "student_id": student_id})
