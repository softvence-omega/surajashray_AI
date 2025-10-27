from doctr.models import ocr_predictor, db_resnet50, crnn_mobilenet_v3_large
from doctr.io import DocumentFile

from app.utils.logger import get_logger
from app.config import DETECTION_MODEL, RECOGNIZE_MODEL

logger = get_logger(__name__)


class OCR:
    
    def __init__(self):
        self.detection_model = db_resnet50(pretrained=False, pretrained_backbone = False)
        self.detection_model.from_pretrained(path_or_url=DETECTION_MODEL, map_location = False)

        self.recognize_model = crnn_mobilenet_v3_large(pretrained=False, pretrained_backbone = False) 
        self.recognize_model.from_pretrained(path_or_url=RECOGNIZE_MODEL, map_location = False)
    
    @staticmethod
    def format_text(text) -> str:
        all_text = []
        
        for page in text["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    line_text = " ".join([word["value"] for word in line["words"]])
                    all_text.append(line_text)
                    
        final_text = " ".join(all_text)
        return final_text
    
    # for pdf file
    def pdf_to_text(self, pdf_file) -> str:
        
        try:
            logger.info("OCR Model Initializing For PDF to Text..........")
            self.model = ocr_predictor(det_arch=self.detection_model,reco_arch=self.recognize_model,pretrained=True)
            self.pdf = DocumentFile.from_pdf(pdf_file)
            self.result = self.model(self.pdf)
            self.result = self.result.export()

            self.final_text = self.format_text(self.result)
            
            return self.final_text
        except Exception as e:
            raise ValueError(e)
        
    # for img file
    def img_to_text(self, img_file) -> str:     
        try:
            logger.info("OCR Model Initializing For Img to Text..........")
            self.model = ocr_predictor(det_arch=self.detection_model,reco_arch=self.recognize_model,pretrained=True)
            self.img = DocumentFile.from_images(img_file)
            self.result = self.model(self.img)
            self.result = self.result.export()

            self.final_text = self.format_text(self.result)
            
            return self.final_text
        except Exception as e:
            raise ValueError(e)
        
        


