# RallyClipper

This repository handles the automatic clipping of long (~1 hour) raw table tennis match videos into shorter clips of individual rallies.

The training data can be download through
```bash
wget https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/resolve/main/RallyClipper.zip
unzip RallyClipper.zip
cp -r RallyClipper/* .
rm -r RallyClipper
rm RallyClipper.zip
```