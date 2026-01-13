from pydantic import BaseModel, Field
from typing import Literal, Optional, List, TypedDict

# ==================== SCHEMAS ====================
class MedicalReportClassify(BaseModel):
    check: Literal["yes", "no"] = Field(..., description="Check if the text/image is a medical report (ECG, blood test, X-ray, CT scan, lab report, etc.) or not. Look for medical terminology, patient info, test results, or medical charts/graphs.")

class LabValue(BaseModel):
    test_name: str = Field(..., description="Name of the lab test or observation")
    value: Optional[float] = Field(None, description="Measured value from lab report, if applicable")
    unit: Optional[str] = Field(None, description="Unit of the measured value, if applicable")
    ref_ranges: Optional[str] = Field(None, description="Standard clinical reference range, if applicable")
    status: Optional[str] = Field(None, description="Whether the value is 'within', 'above', or 'below' the normal range, if applicable")

class WellnessReport(BaseModel):
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    report_date: Optional[str] = Field(None, description="Date of the lab report")
    lab_values: List[LabValue] = Field(..., description="List of lab values or observations with wellness insights")
    wellness_insight: str = Field(..., description="Non-medical, lifestyle-oriented advice based on the value or test")

class ReportState(TypedDict):
    """Represent the structure of the state used in graph"""
    report_text: str
    report_status: str
    output: str