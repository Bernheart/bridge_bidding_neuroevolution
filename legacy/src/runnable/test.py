old = b"evolution_agent"
new = b"src.evolution.evolution_agent"

with open("./archive/v0.0.0/population.pkl", "rb") as f:
    data = f.read()

if old in data:
    # Replace the bytes
    patched = data.replace(old, new)

    # Save the fixed version
    with open("your_file_fixed.pkl", "wb") as f:
        f.write(patched)

    print("🔧 Patched pickle file (binary hack).")
else:
    print("⚠️ Old module path not found.")