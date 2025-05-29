#!/usr/bin/env python3
"""
Tinfoil Proxy Server
An OpenAI-compatible API server that routes requests to Tinfoil's secure enclave models
"""

import os
import sys
import json
import logging
from typing import Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tinfoil import TinfoilAI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if getattr(sys, 'frozen', False) else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model mapping configuration
MODEL_CONFIGS = {
    "deepseek-r1-70b": {
        "model_id": "deepseek-r1-70b",
        "description": "High-performance reasoning model",
        "enclave": "deepseek-r1-70b-p.model.tinfoil.sh",
        "repo": "tinfoilsh/confidential-deepseek-r1-70b-prod"
    },
    "llama3-3-70b": {
        "model_id": "llama3-3-70b",
        "description": "Multilingual model optimized for dialogue",
        "enclave": "llama3-3-70b.model.tinfoil.sh",
        "repo": "tinfoilsh/confidential-llama3-3-70b"
    },
    "nomic-embed-text": {
        "model_id": "nomic-embed-text",
        "description": "Text embedding model",
        "enclave": "nomic-embed-text.model.tinfoil.sh",
        "repo": "tinfoilsh/confidential-nomic-embed-text"
    }
}

# OpenAI API request/response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: Optional[bool] = Field(default=True)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    presence_penalty: Optional[float] = Field(default=0.0)
    n: Optional[int] = Field(default=1)
    stop: Optional[list[str]] = Field(default=None)
    stream_options: Optional[dict] = Field(default=None)

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "tinfoil"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

class TinfoilProxyServer:
    def __init__(self):
        self.api_key = os.getenv("TINFOIL_API_KEY")
        if not self.api_key:
            raise ValueError("TINFOIL_API_KEY environment variable is required")
        
        # Initialize Tinfoil clients for each model
        self.clients: Dict[str, TinfoilAI] = {}
        
        for model_name, config in MODEL_CONFIGS.items():
            try:
                # Debug info for attestation
                import traceback
                logger.debug(f"Initializing {model_name}")
                logger.debug(f"Python frozen: {getattr(sys, 'frozen', False)}")
                logger.debug(f"Bundle dir: {getattr(sys, '_MEIPASS', None)}")
                
                self.clients[model_name] = TinfoilAI(
                    api_key=self.api_key,
                    enclave=config["enclave"],
                    repo=config["repo"]
                )
                logger.info(f"Successfully initialized Tinfoil client for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                logger.debug(f"Full error: {traceback.format_exc()}")
                # Continue with other models instead of failing completely
                logger.warning(f"Skipping model {model_name} due to initialization error")

    def get_client(self, model: str) -> TinfoilAI:
        """Get the appropriate Tinfoil client for the given model"""
        if model not in self.clients:
            raise HTTPException(status_code=400, detail=f"Model '{model}' not supported")
        return self.clients[model]

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Stream chat completion responses in SSE format"""
        client = self.get_client(request.model)
        
        try:
            # Convert messages to the format expected by Tinfoil
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Create streaming chat completion
            stream = client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                n=request.n,
                stop=request.stop,
            )
            
            # Stream responses in SSE format
            for chunk in stream:
                # Convert to OpenAI-compatible format
                chunk_data = {
                    "id": chunk.id,
                    "object": "chat.completion.chunk",
                    "created": chunk.created,
                    "model": request.model,
                    "choices": []
                }
                
                for choice in chunk.choices:
                    choice_data = {
                        "index": choice.index,
                        "delta": {},
                        "finish_reason": choice.finish_reason
                    }
                    
                    if hasattr(choice.delta, 'role'):
                        choice_data["delta"]["role"] = choice.delta.role
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        choice_data["delta"]["content"] = choice.delta.content
                    
                    chunk_data["choices"].append(choice_data)
                
                # Include usage data if requested
                if request.stream_options and request.stream_options.get("include_usage"):
                    if hasattr(chunk, 'usage') and chunk.usage:
                        chunk_data["usage"] = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final [DONE] message
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def non_streaming_chat_completion(self, request: ChatCompletionRequest) -> dict:
        """Handle non-streaming chat completion requests"""
        client = self.get_client(request.model)
        
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            response = client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=False,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                n=request.n,
                stop=request.stop,
            )
            
            # Convert to OpenAI-compatible format
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": request.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error during non-streaming completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize proxy server
proxy_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global proxy_server
    proxy_server = TinfoilProxyServer()
    logger.info("Tinfoil proxy server initialized")
    yield
    logger.info("Tinfoil proxy server shutting down")

# Create FastAPI app
app = FastAPI(
    title="Tinfoil Proxy",
    description="OpenAI-compatible API proxy for Tinfoil secure enclave models",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models"""
    global proxy_server
    
    if not proxy_server:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    models = []
    # Only list models that are actually initialized
    for model_id in proxy_server.clients.keys():
        models.append(ModelInfo(id=model_id))
    return ModelsResponse(data=models)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """Handle chat completion requests"""
    global proxy_server
    
    if not proxy_server:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Log request details
    logger.info(f"Chat completion request for model: {request.model}, streaming: {request.stream}")
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            proxy_server.stream_chat_completion(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )
    else:
        # Return non-streaming response
        return await proxy_server.non_streaming_chat_completion(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tinfoil-proxy"}

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("TINFOIL_PROXY_PORT", "8093"))
    # When running as a binary, we need to use app directly (not module:app string)
    # and disable reload to avoid multiple processes
    import sys
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller binary
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
    else:
        # Running as Python script
        uvicorn.run(
            "tinfoil_proxy:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info"
        )