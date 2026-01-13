# WellnessTrack

WellnessTrack is an AI-powered FastAPI application that provides comprehensive wellness insights through advanced document analysis and nutritional scanning capabilities.

## Features

- **Lab Report Analysis**: Upload lab reports in PDF or image formats to extract health data using OCR and generate AI-powered lifestyle recommendations
- **Meal Scanning**: Upload meal images to analyze nutritional content using computer vision and AI models
- **AI-Powered Insights**: Leverages LangChain, LangGraph, and OpenAI GPT for intelligent analysis
- **Cloud Integration**: Automatic file uploads to Cloudinary for processed documents
- **Real-time Processing**: Fast, efficient analysis with background task management

## Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: LangChain, LangGraph, OpenAI GPT
- **OCR**: Custom OCR pipeline for document text extraction
- **Computer Vision**: Advanced models for meal analysis
- **Cloud Storage**: Cloudinary integration
- **Validation**: Pydantic schemas for type-safe API interactions

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd WellnessTrack
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file in the project root
   - Add your API keys:

     ```
     # OpenAI Configuration
     OPENAI_API_KEY=your_openai_api_key_here

     # Cloudinary Configuration (for file uploads)
     CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
     CLOUDINARY_API_KEY=your_cloudinary_api_key
     CLOUDINARY_API_SECRET=your_cloudinary_api_secret
     ```

## Usage

Run the FastAPI application:

```bash
python main.py
```

The API will be available at `http://127.0.0.1:8080`.

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://127.0.0.1:8080/docs`
- **ReDoc**: `http://127.0.0.1:8080/redoc`

### API Endpoints

#### Lab Report Analysis

- **POST** `/lab_report_analysis`
- **Description**: Upload and analyze lab reports (PDF or images) using OCR and AI
- **Content-Type**: `multipart/form-data`
- **Request**: File upload with field name `file`
- **Supported Formats**: PDF, JPEG, PNG, BMP
- **Response**:
  ```json
  {
    "file_name": "report.pdf",
    "report_text": {
      "analysis": "AI-generated wellness insights...",
      "recommendations": "Personalized health recommendations..."
    },
    "file_url": "https://cloudinary.com/uploaded-file-url"
  }
  ```

#### Meal Analysis

- **POST** `/api/v1/analyze-food`
- **Description**: Analyze meal images for nutritional content
- **Content-Type**: `multipart/form-data`
- **Request**: File upload with field name `file`
- **Response**:
  ```json
  {
    "nutrition": {
      "calories": 450,
      "protein_g": 25,
      "carbs_g": 30,
      "fats_g": 20,
      "is_meal": true
    }
  }
  ```

## Project Structure

```
WellnessTrack/
├── main.py                    # Application entry point
├── fast-api_structure.py      # API structure definitions
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── config/config.yaml        # Application configuration
├── app/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── api/v1/
│   │   ├── __init__.py
│   │   └── endpoints/        # API route handlers
│   │       ├── lab_report.py     # Lab report analysis endpoint
│   │       └── meal_scaner.py    # Meal analysis endpoint
│   ├── model/                # AI/ML models
│   │   ├── crnn_mobilenet_v3_large_pt.pt
│   │   └── db_resnet50.pt
│   ├── schemas/              # Pydantic data models
│   │   ├── meal_evaluation.py
│   │   ├── mealstate.py
│   │   └── schema.py
│   ├── services/             # Business logic layer
│   │   ├── meal_analyzer.py
│   │   ├── meal_graph.py
│   │   └── lab_report/       # Lab report processing services
│   │       ├── graph/        # LangGraph workflows
│   │       ├── llms/         # LLM integrations
│   │       ├── nodes/        # Processing nodes
│   │       ├── ocr/          # OCR services
│   │       └── state/        # State management
│   └── utils/                # Utility functions
│       ├── helper.py         # File operations, cleanup
│       └── logger.py         # Logging configuration
├── logs/                     # Application logs
└── config/                   # Configuration files
```

## Development

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Cloudinary account (optional, for file uploads)

### Local Development

1. Follow the installation steps above
2. Run the application in development mode:
   ```bash
   python main.py
   ```
3. Access the API documentation at `http://127.0.0.1:8080/docs`

## License

[Specify your license here]
