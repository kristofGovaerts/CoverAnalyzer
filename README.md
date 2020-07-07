# CoverAnalyzer
Automatic tool for assessing cover in drone images. This tool performs several functions:
- Calculate global cover per image (in %)
- Count the rows
- Estimate cover per row (in %)
- Generate an index for the amount of gaps in each row
- Generate output:
  - Output annotated image
  - Output table 
  
# Usage
Run */code/cover_analysis.py* and provide a folder containing the input .JPG files. The script will iterate over each .JPG file and generate output accordingly. 

# How it works
This script makes use of the fact that the green channel in RGB drone images of beet fields is hyperintense relative to the red and blue channels. Rows are automatically detected based on the intensity profile along the Y-axis.

## Input file.
<img src="/examples/test_img.JPG" height="300">

The input file is a .JPG RGB image, although other file formats may be supported in the future if there is demand for this. Homogenous lighting and high contrast between leaves and soil will provide optimal results. Perfect lighting conditions are not necessary but the better they are the better the results will be.

## Processing.

The intensity of the green channel relative to the other two channels is calculated using the following formula: 

<img src="/examples/plots/eq.gif">

This allows us to calculate a map in which high values (e.g. above 1) indicate 'green' pixels. An alternative approach could use the H channel in a HSV image, but this method provided nicer results. Important is that pixel sizes are either known or consistent, as rows are calculated based on a fixed row width.

<img src="/examples/plots/fig1.png" height="300"> <img src="/examples/plots/fig2.png" height="300">
 
By calculating the intensity profile along the Y-axis (i.e. averaging every X-axis value for each Y-axis coordinate), a periodic peak pattern is produced in which peaks correspond to the center of rows. Each row is then isolated based on the provided row width (global variable ROW_WIDTH, default 120px). Mean cover (% green pixels) is calculated across the entire image, as well as for each row individually. Furthermore, gaps are identified by taking the mean intensity profile along each row and using a threshold approach to determine gap pixels.
 
## Output
<img src="/examples/test/test_img.png" height="500">

An annotated image is generated for each input .JPG file (assuming none triggered an error). This image shows a **grayscale** image overlaid with the segmented leaves (in transparent yellow). Row numbering is indicated on the side, along with corresponding cover and gap values. 

## Author

© **Kristof Govaerts, PhD** - *[kristof.govaerts@sesvanderhave.com]*

© **SESVanderhave n.v.** - *Industriepark 15, 3300 Tienen* 
