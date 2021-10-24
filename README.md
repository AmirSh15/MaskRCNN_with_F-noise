# Mask-RCNN with F-noise

This module contains code in support of F-noise idea for improving the accuracy of Mask-RCNN network.

In this repository, I used the original implementation of Mask-RCNN from [detectron2](https://github.com/facebookresearch/detectron2) library.
All the experiments are done inside the _Mask-rcnn with F-noise.ipynb_ notebook. (Please check here for results)

### Dependencies

The code was successfully built and run with these versions:

```
detectron2-0.5-cu111
toch 1.9.0-cu11
```
Note: You can also create the environment I've tested with by importing _environment.yml_ in conda.

### Data

For this module we only use a part of coco database with the below distribution in classes.

|   category    | #instances   |   category   | #instances   |   category    | #instances   |
|:-------------:|:-------------|:------------:|:-------------|:-------------:|:-------------|
|    person     | 10777        |   bicycle    | 314          |      car      | 1918         |
|  motorcycle   | 367          |   airplane   | 143          |      bus      | 283          |
|     train     | 190          |    truck     | 414          |     boat      | 424          |
| traffic light | 634          | fire hydrant | 101          |   stop sign   | 75           |
| parking meter | 60           |    bench     | 411          |     bird      | 427          |
|      cat      | 202          |     dog      | 218          |     horse     | 272          |
|     sheep     | 354          |     cow      | 372          |   elephant    | 252          |
|     bear      | 71           |    zebra     | 266          |    giraffe    | 232          |
|   backpack    | 371          |   umbrella   | 407          |    handbag    | 540          |
|      tie      | 252          |   suitcase   | 299          |    frisbee    | 115          |
|     skis      | 241          |  snowboard   | 69           |  sports ball  | 260          |
|     kite      | 327          | baseball bat | 145          | baseball gl.. | 148          |
|  skateboard   | 179          |  surfboard   | 267          | tennis racket | 225          |
|    bottle     | 1013         |  wine glass  | 341          |      cup      | 895          |
|     fork      | 215          |    knife     | 325          |     spoon     | 253          |
|     bowl      | 623          |    banana    | 370          |     apple     | 236          |
|   sandwich    | 177          |    orange    | 285          |   broccoli    | 312          |
|    carrot     | 365          |   hot dog    | 125          |     pizza     | 284          |
|     donut     | 328          |     cake     | 310          |     chair     | 1771         |
|     couch     | 261          | potted plant | 342          |      bed      | 163          |
| dining table  | 695          |    toilet    | 179          |      tv       | 288          |
|    laptop     | 231          |    mouse     | 106          |    remote     | 283          |
|   keyboard    | 153          |  cell phone  | 262          |   microwave   | 55           |
|     oven      | 143          |   toaster    | 9            |     sink      | 225          |
| refrigerator  | 126          |     book     | 1129         |     clock     | 267          |
|     vase      | 274          |   scissors   | 36           |  teddy bear   | 190          |
|  hair drier   | 11           |  toothbrush  | 57           |               |              |
|     total     | 36335        |              |              |               |              |

You can replicate the results with putting the data in a similar directory tree as below:

```
/dataset/
  coco/
     annotations/
        instances.json
     data/
        000000000139.jpg
        ...
```

### References

If you found this repo useful give me a star!