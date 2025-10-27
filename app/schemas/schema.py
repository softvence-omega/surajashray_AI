from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class MedicalReportClassify(BaseModel):
    check : Literal["yes", "no"] = Field(..., description="Check if the text is a medical report or not")
        
    
class LabValue(BaseModel):
    test_name : str = Field(..., description="Name of the lab test or observation")
    value : Optional[float] = Field(None, description="Measured value from lab report, if applicable")
    unit : Optional[str] = Field(None, description="Unit of the measured value, if applicable")
    ref_ranges : Optional[str] = Field(None, description="Standard clinical reference range, if applicable")
    status : Optional[str] = Field(None, description="Whether the value is 'within', 'above', or 'below' the normal range, if applicable")
    
    
class WellnessReport(BaseModel):
    patient_name : Optional[str] = Field(None, description="Name of the patient")
    report_date : Optional[str] = Field(None, description="Date of the lab report")
    lab_values : List[LabValue] = Field(..., description="List of lab values or observations with wellness insights")
    wellness_insight : str = Field(..., description="Non-medical, lifestyle-oriented advice based on the value or test")
    
