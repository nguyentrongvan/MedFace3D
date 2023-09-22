# Dense-face-base-on-3D-landmarks
#### Using medipipe face mesh (468 facial landmarks) to reconstruct 3D face mesh
<div style="text-align:center;">
    <img src="https://mohamedalirashad.github.io/FreeFaceMoCap/assets/img/mediapipe.png" alt="MediaPipe" width="30%">
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
python reconstruct.py
```

## 2. Demo result
### Face mesh generation:
![Demo face mesh](data/demo/dense.png)

### Face depth map estimation:
![Demo face depth](data/demo/depth.png)

### Video face dense estimation:

![](data/demo/famed01.gif)
![](data/demo/famed02.gif)

### 3D face mesh reconstruction:
<video controls>
  <source src="data/demo/mesh_famed01.mp4" type="video/mp4">
</video>

<video controls>
  <source src="data/demo/mesh_local.mp4" type="video/mp4">
</video>

