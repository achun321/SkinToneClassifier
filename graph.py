import matplotlib.pyplot as plt
import numpy as np

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

# Replace these hex color codes with the colors you want to compare
hex_color1 = "#755542"
hex_color2 = "#AE846F"

color1 = hex_to_rgb(hex_color1)
color2 = hex_to_rgb(hex_color2)

distance = color_distance(color1, color2)

plt.figure(figsize=(6, 6))
plt.scatter([color1[0], color2[0]], [color1[1], color2[1]], c=[hex_color1, hex_color2], s=300)  # Adjust dot size
plt.xlabel('Red')
plt.ylabel('Green')
plt.title(f'Color Distance: {distance:.2f}')

# Add a dotted line connecting the two colors
plt.plot([color1[0], color2[0]], [color1[1], color2[1]], linestyle='dotted', color='gray')
plt.xlim(min(color1[0], color2[0]) - 10, max(color1[0], color2[0]) + 10)
plt.grid(False)  # Turn off gridlines
plt.show()