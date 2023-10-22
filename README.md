# Disaster Response Mapping
Submission for IBM Data Z Datathon 2023. Link to devpost [here](https://devpost.com/software/disaster-response-mapping?ref_content=my-projects-tab&ref_feature=my_projects).

## About
Aimed to provide real-time, accurate and useful road network data to improve the timeliness of humanitarian aid, Our program takes an input of a recent satellite image and outputs the same image with an overlay of all usable roads with their gradients colour coded for easy reference. 

Our quick generation maps help to reduce "surprises" on the ground, such as unexpected obstructions, so that humanitarian aid planners can quickly deploy their resources. We also chose to include gradient data, as steep slopes can prove difficult to traverse, as there may even be risks of landslides under rainy conditions.

## Usage

To use, `cd` to the working directory with `main.py`.

Then run the file in terminal with the following argument:

`python main.py --filepath "path-to-input.tif" --output_dir "path-to-output-dir" --dtm_path "path-to-dtm"`

#### Example Input:
![Input](/sample_imgs/input_zoom.jpg)

#### Output:
![Output](/sample_imgs/output_zoom.png)

Click [here](https://mega.nz/fm/I3lwCQ5L) for raw sample input/output files
