import os
import cv2
import matplotlib.pyplot as plt

# ✅ Correct path (based on your folder tree)
path = r"data/test/plant_diseases/PlantVillage/Tomato_healthy"

# Check if path exists
if not os.path.exists(path):
    print("❌ Path not found:", path)
else:
    print("✅ Path found:", path)

    # Get one image from the folder
    files = os.listdir(path)
    if len(files) == 0:
        print("⚠️ No images found in the folder.")
    else:
        img_path = os.path.join(path, files[0])
        print("📸 Displaying:", img_path)

        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Sample Image - Tomato Healthy")
        plt.axis('off')
        plt.show()
