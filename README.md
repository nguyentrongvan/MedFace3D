# Dense-face-base-on-3D-landmarks
#### Using medipipe face mesh to detect 468 3D facial landmarks and use it to reconstruct 3D face mesh
<div style="text-align:center;">
    <img src="https://mohamedalirashad.github.io/FreeFaceMoCap/assets/img/mediapipe.png" alt="MediaPipe" width="30%">
</div>
<div style="text-align:center;">
    <img src="data/sample/face_001.jpg_face_reconstruction.gif" alt="demo" width="30%">
</div>


## 1. Usage
***1.1. Setup env***  
```
git clone https://github.com/nguyentrongvan/Dense-face-base-on-3D-landmarks.git
cd Dense-face-base-on-3D-landmarks 
pip install -r requirements.txt
```


***1.2. Run demo***  
```
python demo.py
```

## 2. Demo result
### Face dense generation:
![Demo face mesh](data/demo/dense.png)
<div style="text-align:center;">
    <img src="data/demo/famed01.gif" alt="MediaPipe" width="49%">
    <img src="data/demo/famed02.gif" alt="MediaPipe" width="49%">
</div>


### Face depth estimation:
![Demo face depth](data/demo/depth.png)


### 3D face mesh reconstruction:
<div style="text-align:center;">
    <img src="data/demo/mesh_famed.gif" alt="MediaPipe" width="100%">
    <img src="data/demo/mesh_local.gif" alt="MediaPipe" width="100%">
</div>

