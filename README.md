# Face Recognition System

A real-time face recognition system that detects face orientation (left, front, right) and recognizes registered individuals using webcam input.

## Features

- **Real-time Face Detection**: Detects faces and determines orientation (LEFT, FRONT, RIGHT)
- **Face Recognition**: Identifies registered individuals and displays their name
- **Multi-angle Registration**: Register new people by capturing their face from three angles (similar to Apple's Face ID)
- **Persistent Database**: Saves registered faces between sessions

## Requirements

- Python 3.11
- Webcam
- macOS, Linux, or Windows

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MajdHail/FaceRecognition.git
cd FaceRecognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python src/main.py
```

### Keyboard Controls

- **`q`** - Quit the program
- **`r`** - Register a new person
- **`d`** - Delete all registered faces (requires confirmation)
- **`l`** - List all registered people in the console

### Registering a New Person

1. Press `r` while the program is running
2. Enter the person's name in the console
3. Follow the on-screen instructions:
   - Turn your head **LEFT**
   - Look straight **FRONT**
   - Turn your head **RIGHT**
4. The system automatically captures each angle when you're in position
5. Once complete, the person is saved to the database

### Recognition

Once registered, the system will:
- Automatically recognize the person when they appear on camera
- Display their name in the **top right corner**
- Continue showing face orientation (LEFT, FRONT, RIGHT) in the **top left**

## How It Works

The system uses:
- **MediaPipe Face Mesh** for facial landmark detection
- **Landmark-based embeddings** for face recognition
- **Euclidean distance** to match faces against the database
- **Multi-angle capture** to improve recognition accuracy from different perspectives

## Database

Registered faces are stored in `face_database.pkl` in the same directory as the script. This file is automatically created when you register your first person.

⚠️ **Note**: The database file contains facial recognition data. Keep it secure and do not commit it to public repositories.

## Troubleshooting

**Webcam not working:**
- Make sure your webcam is connected and not being used by another application
- On macOS, grant camera permissions when prompted
- Try changing `cv2.CAP_AVFOUNDATION` to `0` in the code if you're on Windows/Linux

**Recognition not working well:**
- Ensure good lighting when registering faces
- Stand at a similar distance during registration and recognition
- Re-register with better quality captures if needed

**"Could not open webcam" error:**
- Check that your webcam is connected
- Try changing the camera index from `0` to `1` or `2` if you have multiple cameras

## License

MIT License - feel free to use and modify as needed.
