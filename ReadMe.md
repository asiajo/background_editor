# Background editor

Created for testing the capabilities of background extraction from the photos with a person by Tiramisu
model: <a>https://arxiv.org/abs/1611.09326</a>
It takes images from specified folder, extracts the person from the photos and than smooths, reduces saturation and make the background brighter.

## Usage:
<pre>python3 background_edit.py</pre>
or with optional arguments (which in shown case are equal do default ones):
<pre>python3 background_edit.py --photos_folder ./sample --photos_format jpg --output_folder out</pre>

## Samples:
<p align="center">
<img src="https://user-images.githubusercontent.com/25400249/81480886-c89a6780-922c-11ea-90e8-5ebb770c0909.jpg" width="300" />
<img src="https://user-images.githubusercontent.com/25400249/81480894-d354fc80-922c-11ea-95cb-f9eeb6e1e6e7.png" width="300" /><br>
<img src="https://user-images.githubusercontent.com/25400249/81480889-cc2dee80-922c-11ea-95cc-ff00f511f3db.jpg" width="300" />
<img src="https://user-images.githubusercontent.com/25400249/81480898-d7811a00-922c-11ea-970d-9398aa2a22af.png" width="300" /><br>
<img src="https://user-images.githubusercontent.com/25400249/81480892-cf28df00-922c-11ea-9522-8a37af44954b.jpg" width="300" />
<img src="https://user-images.githubusercontent.com/25400249/81480902-da7c0a80-922c-11ea-8904-bdbcd0e3578c.png" width="300" />
</p>


