import asyncio
import aiohttp
import json
import logging # Added
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable, Type # Added Type
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore