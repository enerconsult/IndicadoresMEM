from PIL import Image
import os

try:
    img_path = "logo_empresa.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        print(f"Original size: {img.size}, {os.path.getsize(img_path)/1024:.2f} KB")
        
        # Resize maintaining aspect ratio (Width 200px is enough for sidebar)
        base_width = 200
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        
        # Save optimized
        img.save("logo_empresa_opt.png", optimize=True, quality=85)
        print(f"New size: {img.size}, {os.path.getsize('logo_empresa_opt.png')/1024:.2f} KB")
        
        # Replace original
        os.replace("logo_empresa_opt.png", "logo_empresa.png")
        print("Logo replaced with optimized version.")
    else:
        print("Logo not found.")
except Exception as e:
    print(f"Error optimizing image: {e}")
