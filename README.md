# Vision Lab

A desktop hub for small webcam-based computer-vision projects. The main window shows a card
for each module; click a card to open it, and use **← Back to Hub** to return (this also turns
the webcam off).

Modules so far:

- **Hello**: minimal example that shows the live camera feed. Use it as a template.
- **Face Recognition**: enroll people with the webcam and recognize them live
  (OpenCV YuNet for detection, SFace for recognition).

All face data stays on your computer. The only network access is the one-time model download.

## Setup

You need **Python 3.11** and a webcam. Everything is installed into a virtual environment
(`.venv`) inside the project folder; never use the global Python or pip for this project.

### Windows (PowerShell)

```powershell
cd D:\projects\vision_lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

If `Activate.ps1` is blocked, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
You can also skip activation and call the venv directly: `.\.venv\Scripts\python.exe main.py`.

### Linux

```bash
cd vision_lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

On Debian/Ubuntu, Qt also needs `sudo apt install libxcb-cursor0`. If the camera can't be
opened, make sure your user is in the `video` group.

### Next time

Activate the venv (`.\.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`) and run
`python main.py`.

## Using Face Recognition

**Enroll tab**

1. Type the person's name.
2. Click **Capture** for single samples, or turn on **Auto-capture** to take one every 0.5 s.
   While capturing, slowly turn your head a little so the samples cover different angles.
3. Exactly one face must be visible; otherwise the sample is rejected with a warning
   (the boxes turn red when more than one face is visible).
4. The target is 15 samples. Enrolling the same name again adds samples to that person.

The list shows everyone enrolled; **Delete** removes a person's images and embeddings.

**Recognize tab**

- Each face gets the name of the best match and its score, or **Unknown** if the score is
  below the **match threshold**.
- The threshold is a cosine similarity between 0 and 1 (default 0.363, OpenCV's recommended
  value for SFace). Raise it if strangers get named; lower it if enrolled people show as Unknown.
- **Recognize every N frames** saves CPU: in between, the last labels are kept on screen.
- The FPS line shows the video rate and how many recognitions run per second.

**Where the data lives**

```
data/faces/<name>/0001.jpg ...   aligned face crops (112x112)
data/embeddings.npz              128-number face embeddings per person
data/people.json                 folder name -> name as typed
```

Folder names are simplified versions of the typed name (for example `José Pérez` →
`jose_perez`); the original name is kept for display. `data/` and `models/` are not committed to git.

## Troubleshooting

| What you see | What to do |
|---|---|
| "Face models not found" | Run `python scripts/download_models.py` (with the venv active). |
| "Could not load the face models" | A model file is damaged. Delete the `models` folder and run the download script again. |
| "No camera found" | Connect a webcam, close other apps that use it (Zoom, Teams, the Camera app), check Windows camera privacy settings, then go back to the hub and open the module again. |
| "Camera disconnected" | Reconnect the camera, go back to the hub and open the module again. |
| "Can't write to …\data" | The folder is read-only or blocked. Fix its permissions; recognition keeps working meanwhile. |
| "Could not read the saved faces" | `data/embeddings.npz` or `data/people.json` is damaged. Move them out of `data/` (or restore a backup) and reopen. |
| A grey "failed to load" card | That module's code raised an error on import. Hover the card for the error; the full traceback is in the terminal. |
| VS Code: `Import "PySide6" could not be resolved` | Command Palette → **Python: Select Interpreter** → choose `.venv`. |

## Project structure

```
main.py                     starts the app and the shared camera
core/
  hub_window.py             main window: card grid + module pages
  base_module.py            BaseModule, the interface every module implements
  module_registry.py        finds modules/*/module.py automatically
  camera.py                 shared Camera (QThread) -> frame_ready(np.ndarray), error(str)
  utils.py                  frame -> QImage, image read/write, UiCallback
modules/
  hello/                    minimal reference module
  face_recognition/         module.py (UI), detector.py, recognizer.py, storage.py, worker.py
scripts/download_models.py  downloads the ONNX models from the OpenCV Zoo
```

## How to add a new module

A module is a folder in `modules/` with a `module.py` that defines one `BaseModule` subclass.
The hub finds it automatically. Nothing else needs to change.

### 1. Copy the Hello module

```
modules/hello/   ->   modules/my_module/
```

Use lowercase letters and underscores for the folder name (it becomes a Python package).
The folder needs `__init__.py` (empty) and `module.py`.

### 2. Edit `module.py`

Here is the Hello module, shortened:

```python
class HelloModule(BaseModule):
    name = "Hello"                                     # card title
    description = "Minimal example module with a live camera feed."
    # icon = "icon.png"                                # optional, relative to this folder

    def __init__(self, camera):
        super().__init__(camera)
        self._active = False

    def create_widget(self, parent):                   # called once; the result is cached
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        self.video = QLabel()
        ...
        return widget

    def on_open(self):                                 # the user opened the module
        self._active = True
        self.camera.frame_ready.connect(self._show_frame)
        self.camera.error.connect(self._show_error)
        self.camera.start()

    def on_close(self):                                # Back to Hub, or the window closed
        self._active = False
        self.camera.frame_ready.disconnect(self._show_frame)
        self.camera.error.disconnect(self._show_error)
        self.camera.stop()                             # releases the webcam

    def _show_frame(self, frame):                      # frame is a BGR numpy array
        ...
```

Rename the class, set `name` and `description`, and build your UI in `create_widget`.

### 3. Follow the rules

- **Camera:** connect to `frame_ready` and `error` in `on_open` and start the camera; disconnect
  and stop it in `on_close`. Only one module is open at a time, so they share one camera.
- **Always handle `error`** and show the message where the video would be.
- **Keep the UI thread fast.** Anything slower than a few milliseconds per frame (neural networks,
  MediaPipe, etc.) belongs in a worker: a `QObject` moved to its own `QThread`. Send it a frame
  only when the previous result has come back, so frames are dropped instead of piling up.
  `modules/face_recognition/worker.py` is a complete example.
- **Receive worker results with `UiCallback`.** A worker signal connected straight to a plain
  Python method runs that method on the *worker* thread, where touching widgets can crash.
  Instead:

  ```python
  self.result_relay = UiCallback(self._on_result)      # create on the UI thread
  self.worker.result.connect(self.result_relay.call)
  ```

- **Stop your threads in `on_close`** (`thread.quit()` then `thread.wait()`).
- Models go in `models/` (add them to `scripts/download_models.py`); saved data goes in `data/`.
  Read and write images with `core.utils.read_image` / `write_image`, which work with any
  characters in the path.
- New dependencies go in `requirements.txt` with an exact version, installed into `.venv`.

### 4. Run it

Restart the app; the new card appears. If your module fails to import, it shows as a grey
"failed to load" card (hover it for the error) and the rest of the hub keeps working.
