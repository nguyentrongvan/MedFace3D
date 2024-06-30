# medface3D.worker
### Dense face reconstruction based on 3D facial landmarks
#### Using medipipe face mesh to detect 468 3D facial landmarks and use it to reconstruct 3D face mesh
<div style="text-align:center;">
    <img src="https://mohamedalirashad.github.io/FreeFaceMoCap/assets/img/mediapipe.png" alt="MediaPipe" width="30%">
</div>
<div style="text-align:center;">
    <img src="https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/sample/face_001.jpg_face_reconstruction.gif?raw=true" alt="demo" width="30%">
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
![Demo face mesh](https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/dense.png?raw=true)
<div style="text-align:center;">
    <img src="https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/famed01.gif?raw=true" alt="MediaPipe" width="49%">
    <img src="https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/famed02.gif?raw=true" alt="MediaPipe" width="49%">
</div>


### Face depth estimation:
![Demo face depth](https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/depth.png?raw=true)


### 3D face mesh reconstruction:
<div style="text-align:center;">
    <img src="https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/mesh_famed.gif?raw=true" alt="MediaPipe" width="100%">
    <img src="https://github.com/nguyentrongvan/MedFace3D/blob/main/medface3D/data/demo/mesh_local.gif?raw=true" alt="MediaPipe" width="100%">
</div>

