import numpy as np
import pyaudio
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# run instructions:
# python3 golwsound_freq.py --grid-size 50

# Global variables for sound input
CHUNK = 4096
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]

def randomGrid(N):
    """Returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)
    
def process_sound_input(stream, grid, N):
    # Read raw data
    data = stream.read(CHUNK)
    # Convert data to numpy array
    data_np = np.frombuffer(data, dtype=np.float32)
        
    # Perform FFT and calculate frequencies
    fft_result = np.fft.rfft(data_np)
    freqs = np.fft.rfftfreq(len(data_np), 1/RATE)
        
    # Find the peak frequency
    idx = np.argmax(np.abs(fft_result))
    dominant_freq = freqs[idx]
        
    print(f"{dominant_freq:.2f}")
    
    # Map dominant frequency to grid influence
    probability_threshold = min(1.0, dominant_freq / 2000)  # Example thresholding, adjust as needed

    # Use dominant frequency to influence the grid
    for i in range(N):
        for j in range(N):
            if np.random.rand() < probability_threshold:
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
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_visible(False)
    img = plt.imshow(grid, interpolation='nearest', cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the margins

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
