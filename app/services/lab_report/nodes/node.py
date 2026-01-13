from langchain_core.messages import HumanMessage, SystemMessage
import json

from app.services.lab_report.state.state import ReportState
from app.schemas.schema import MedicalReportClassify, WellnessReport
from app.config import SYSTEM_PROMPT
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ReportNode:
    
    def __init__(self, model):
        self.llm = model
        
    ### Node 1
    def classify_report(self, state: ReportState):
        """
            Determine if the report text is medical and update the state.

            Args:
                state (ReportState): Contains 'report_text' key with the report content.

            Returns:
                ReportState: Updated state with 'report_status' indicating classification.
        """
        print("CLASSIFYING REPORT............")
        
        self.text = state["report_text"]
        
        try:
            self.llm_with_structure_output = self.llm.with_structured_output(MedicalReportClassify)
            output = self.llm_with_structure_output.invoke(self.text)
            
            state["report_status"] = output.check
        
            return state
        except Exception as e:
            raise ValueError(e)
    
    ### Node 2
    def generate_report(self, state: ReportState):
        """
            Generate a structured JSON wellness report from the report text.

            Args:
                state (ReportState): Contains 'report_text' key with the input text.

            Returns:
                ReportState: Updated state with 'output' key containing the JSON report.
        """
        print("GENERATING REPORT............")

        self.input_text = state["report_text"]
        
        try:
        
            self.llm_report_structure = self.llm.with_structured_output(WellnessReport)
            
            self.output = self.llm_report_structure.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content = self.input_text)
            ])
            
            self.final_output = self.output.model_dump_json(indent=4)
            
            state["output"] = self.final_output
            
        except ValueError as e:
            self.fall_back_report = WellnessReport(
                patient_name = None,
                report_date = None,
                lab_values = [],
                wellness_insight = "Unable to generate detailed insights. Please ensure the medical report contains valid test results."
            )
            state["output"] = self.fall_back_report.model_dump_json(indent=4)
            logger.error(f"Validation Error: {e}")
            
        return state
    
    ### Node 3
    def not_report(self, state: ReportState):
        """
            Handle cases where the input text is not a valid medical or lab report.

            Args:
                state (ReportState): The current state of the report processing.

            Returns:
                ReportState: Updated state with 'output' containing an error message.
        """
        print("NOT ACTUAL REPORT............")
        self.data = "The provided text does not appear to be a medical or lab report. Please upload a valid health report for analysis."
        state["output"] = json.dumps(self.data, indent=4)
        
        return state
    
    ### decision function
    def router_decision(self, state: ReportState):
        """
            Decide the next action based on whether the report is medical.

            Args:
                state (ReportState): Contains 'report_status' key indicating classification.

            Returns:
                str: 'generate_report' if medical, otherwise 'not_report'.
        """
        if state['report_status'] == "yes":
            return "generate_report"
        else:
            return "not_report"