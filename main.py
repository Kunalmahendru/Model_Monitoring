from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import dataset_router

app = FastAPI()

#  Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; in production, set to your frontend URL (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(dataset_router.router)
