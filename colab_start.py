"""
Google Colab Launcher for CatVTON Virtual Try-On API
Runs the Flask Backend on Colab GPU and exposes a public HTTPS API endpoint via Cloudflare Tunnel or Ngrok.
"""
import os
import sys
import time
import subprocess
import re
import urllib.request

def check_environment():
    print("=" * 65)
    print("1. Checking Hardware & GPU Acceleration")
    print("=" * 65)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"CUDA GPU Active: {device_name} ({vram_gb} GB VRAM)")
            print("Ready for high-speed CatVTON diffusion inference.")
        else:
            print("WARNING: CUDA is NOT available! Running on CPU will be extremely slow.")
            print("Please go to Colab Menu -> Runtime -> Change runtime type -> Select T4 GPU.")
    except Exception as e:
        print(f"Error checking torch: {e}")

def setup_cloudflare_tunnel(port=5000):
    print("\n" + "=" * 65)
    print("2. Launching Cloudflare Tunnel (Free, No Sign-up Needed)")
    print("=" * 65)
    
    # Download cloudflared binary if not present
    cloudflared_bin = "./cloudflared"
    if not os.path.exists(cloudflared_bin):
        print("Downloading Cloudflare Tunnel CLI...")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        urllib.request.urlretrieve(url, cloudflared_bin)
        os.chmod(cloudflared_bin, 0o777)
        print("Cloudflare CLI ready.")

    # Launch tunnel in background and capture output
    print(f"Connecting local port {port} to global Cloudflare network...")
    tunnel_cmd = [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    tunnel_process = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    public_url = None
    timeout = 30
    start_time = time.time()

    # Search for trycloudflare.com URL in output
    for line in tunnel_process.stdout:
        print("  [Tunnel]", line.strip())
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break
        if time.time() - start_time > timeout:
            break

    if public_url:
        print("\n" + "╔" + "═" * 63 + "╗")
        print(f"║  YOUR PUBLIC API IS LIVE AT:                         ║")
        print(f"║  {public_url.ljust(53)}        ║")
        print("╚" + "═" * 63 + "╝")
        print("\nHow to use in your Next.js project:")
        print(f"1. In ai-tryon/.env.local, set:")
        print(f"   NEXT_PUBLIC_TRYON_API_URL={public_url}")
        print("2. Or paste this URL directly in the Studio Try-On Advanced Settings!")
    else:
        print("Notice: Could not parse Cloudflare URL automatically. Check tunnel log above.")

    return tunnel_process, public_url

def start_flask_server():
    print("\n" + "=" * 65)
    print("3. Starting Flask Virtual Try-On API Server")
    print("=" * 65)
    from app import app
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    check_environment()
    
    # Start Cloudflare tunnel in a background thread
    import threading
    t = threading.Thread(target=setup_cloudflare_tunnel, args=(5000,), daemon=True)
    t.start()
    
    time.sleep(2)
    # Start Flask API
    start_flask_server()
