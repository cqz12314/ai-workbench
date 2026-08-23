from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.files import router as files_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(files_router, tags=["files"])
api_router.include_router(search_router, tags=["search"])
