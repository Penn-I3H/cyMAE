# CyMAE Application

To run locally:

Run: `docker-compose up --build`

The above will create a `data` directory in your root directory locally.

The example copies files from the `INPUT_DIR` directory to the `OUTPUT_DIR` directory. The directories are set in `dev.env` and are defaulted to `/service/data/input` and `/service/data/ouput` for the input and output directories respectively.
To test, create `input` and `output` subfolders in the `data` directory. Create a test file (for example `test.fcs`) in the `/service/data/input` directory. 

Re-Run: `docker-compose up --build`

The png file of predicted outcome should be created in the `data/output` directory.

The current version supports python version 3.10.
