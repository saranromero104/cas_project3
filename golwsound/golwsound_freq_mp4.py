import numpy as np
import pyaudio
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# how to install ffmpeg (Windows powershell as admin):
# install chocolatey: Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
# choco install ffmpeg

# run instructions:
# python3 golwsound_freq_mp4.py --grid-size 50

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

    probability_threshold = min(1.0, dominant_freq / 2000)
    for i in range(N):
        for j in range(N):
            if np.random.rand() < probability_threshold:
                grid[i, j] = ON if grid[i, j] == OFF else OFF

def update(frameNum, img, grid, N, stream):
    process_sound_input(stream, grid, N)
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
    img.set_array(newGrid)
    grid[:] = newGrid[:]
    return img,

def main():
    N = 100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    grid = randomGrid(N)
    
    # Set up matplotlib figure
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_visible(False)
    img = plt.imshow(grid, interpolation='nearest', cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the margins

    # Setup FFMpeg Writer
    writer = FFMpegWriter(fps=20, metadata=dict(title='Game of Life with Sound'), bitrate=1800)

    # Calculate frames for 1 minute
    total_frames = 60 * 1000 // 50  # 60 seconds, 1000 ms/sec, 50 ms interval

    # Set up and save the animation
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, stream), frames=total_frames, blit=True)
    ani.save('golws_test.mp4', writer=writer)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close(fig)

if __name__ == '__main__':
    main()