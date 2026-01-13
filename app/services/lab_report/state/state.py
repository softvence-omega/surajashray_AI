from typing_extensions import TypedDict


class ReportState(TypedDict):
    """
        Represent the structure of the state used in graph
    """
    report_text : str
    report_status : str
    output : str