from fastapi import FastAPI, HTTPException, Depends, UploadFile,File
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from src.utils.config import APP_NAME, VERSION, API_SECRET_KEY
from src.utils.schemas import PredictionResponse
from src.inference import classify_image

# Initialize an app
app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key


@app.get('/', tags=['check'])
async def home(api_key: str=Depends(verify_api_key)):
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "status": "up & running"
    }



@app.post("/classify", tags=['CNN'], response_model=PredictionResponse)
async def classify(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        result = classify_image(contents)
        return PredictionResponse(**result)

    except HTTPException:
        raise  # re-raise custom HTTP exceptions directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

