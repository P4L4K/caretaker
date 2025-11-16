"""
Routes package initialization.
Exports all route modules for the FastAPI application.
"""

# Import all route modules to make them available for import
from . import users
from . import audio
from . import video
from . import unified_stream
from . import cough_stats

# Make these available when importing from routes package directly
__all__ = ['users', 'audio', 'video', 'unified_stream', 'cough_stats']
