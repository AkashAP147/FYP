from tryon_engine import process_tryon
import os

target = r"d:\TRY_ON_MODEL\IMG_20260725_131929.jpg"
source = r"d:\TRY_ON_MODEL\suit-1.webp"

if os.path.exists(target) and os.path.exists(source):
    print(f"Testing with Target: {target}")
    success = process_tryon(target, source, "test_tryon_output.jpg")
    print(f"Result: {'Success' if success else 'Failed'}")
else:
    print(f"Target exists: {os.path.exists(target)}")
    print(f"Source exists: {os.path.exists(source)}")
