import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
import matplotlib.animation as animation

# run instructions:
# python3 golwsound_amp.py --grid-size 50

# Global variables for sound input
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]

def randomGrid(N):
    """Returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

# AddGlider and addGosperGliderGun functions remain the same

# Function to process sound input and modify game behavior
def process_sound_input(stream, grid, N):
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert binary data to integers
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    # Compute the average amplitude of the sound input
    avg_amplitude = np.mean(data_int)
    
    print(f"{avg_amplitude:.2f}")
    # Adjust game behavior based on the average amplitude
    # Example: Increase grid size if average amplitude is high
    # Add your logic here
    # In this example, we'll toggle cells in the grid based on the amplitude
    for i in range(N):
        for j in range(N):
            if np.random.rand() < avg_amplitude / 255:  # Probability based on amplitude
                grid[i, j] = ON if grid[i, j] == OFF else OFF

def update(frameNum, img, grid, N, stream):
    # Process sound input and update the grid
    process_sound_input(stream, grid, N)

    # Compute next generation of cells
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N]) / 255)
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON

    # Update the image with the new grid
    img.set_array(newGrid)
    grid[:] = newGrid[:]
    return img,

def main():
    # Set grid size
    N = 100

    # Set up PyAudio for sound input
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Declare grid
    grid = randomGrid(N)

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')
    plt.colorbar(img)
    plt.axis('off')

    # Set up animation
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, stream),
                                  interval=50, blit=True)

    plt.show()

    # Close the stream and PyAudio instance when done
    stream.stop_stream()
    stream.close()
    p.terminate()

# Call main
if __name__ == '__main__':
    main()
