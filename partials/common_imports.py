# Partials File: partials/common_imports.py
import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore