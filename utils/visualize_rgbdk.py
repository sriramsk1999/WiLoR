import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_path = '/home/jacinto/robot-grasp/third_party/WiLoR/demo_rgbdk/mug_occlusion.npy'
data = np.load(data_path, allow_pickle=True).item()

# Extract the RGB array from the data
rgb_array = data.get('rgb')

# Display the RGB image
plt.figure(figsize=(10,8))
plt.imshow(rgb_array)
plt.axis('off')  # Hide the axis
plt.title('RGB Image from Initial Grasping Scene')
plt.show()

# Display the depth image if available
if 'depth' in data:
    plt.figure(figsize=(10,8))
    plt.imshow(data['depth'], cmap='viridis')
    plt.colorbar(label='Depth (mm)')
    plt.axis('off')
    plt.title('Depth Image from Initial Grasping Scene') 
    plt.show()