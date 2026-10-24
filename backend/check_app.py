import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

print("=" * 60)
print(f" FastAPI Application Title: {app.title}")
print(f" FastAPI Version          : {app.version}")
print(f" Total Routes Configured  : {len(app.routes)}")
print("\nConfigured Endpoint Routes:")
for route in app.routes:
    if hasattr(route, 'path'):
        print(f" -> {route.methods if hasattr(route, 'methods') else 'WS'} {route.path}")
print("=" * 60)
