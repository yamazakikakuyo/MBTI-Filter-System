import os, time, requests
from threading import Thread
import uvicorn
from pyngrok import ngrok

ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])

from app_ori import app as fastapi_app

def run_uvicorn():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, reload=False, workers=1, log_level="info")

server_thread = Thread(target=run_uvicorn, daemon=True)
server_thread.start()

public_tunnel = ngrok.connect(addr=8000, proto="http", bind_tls=True)
base_url = public_tunnel.public_url
print("Public URL:", base_url)
print("Health check URL:", f"{base_url}/health")

for i in range(20):
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        print("Health:", r.status_code, r.json())
        break
    except Exception:
        time.sleep(0.5)
else:
    print("Health check failed to respond.")
